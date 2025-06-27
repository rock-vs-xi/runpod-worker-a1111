import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
import traceback
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from requests.adapters import HTTPAdapter, Retry
from huggingface_hub import HfApi
from schemas.input import INPUT_SCHEMA
from schemas.api import API_SCHEMA
from schemas.img2img import IMG2IMG_SCHEMA
from schemas.txt2img import TXT2IMG_SCHEMA
from schemas.interrogate import INTERROGATE_SCHEMA
from schemas.sync import SYNC_SCHEMA
from schemas.download import DOWNLOAD_SCHEMA

BASE_URI: str = 'http://127.0.0.1:3000'
TIMEOUT: int = 600
POST_RETRIES: int = 3

session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
logger = RunPodLogger()


# ---------------------------------------------------------------------------- #
#                               Utility Functions                              #
# ---------------------------------------------------------------------------- #
def wait_for_service(url: str) -> None:
    """
    Wait for a service to become available by repeatedly polling the URL.

    Args:
        url: The URL to check for service availability.
    """
    retries = 0

    while True:
        try:
            requests.get(url)
            return
        except requests.exceptions.RequestException:
            retries += 1

            # Only log every 15 retries so the logs don't get spammed
            if retries % 15 == 0:
                logger.info('Service not ready yet. Retrying...')
        except Exception as err:
            logger.error(f'Error: {err}')

        time.sleep(0.2)


def send_get_request(endpoint: str) -> requests.Response:
    """
    Send a GET request to the specified endpoint.

    Args:
        endpoint: The API endpoint to send the request to.

    Returns:
        The response from the server.
    """
    return session.get(
        url=f'{BASE_URI}/{endpoint}',
        timeout=TIMEOUT
    )


def send_post_request(endpoint: str, payload: Dict[str, Any], job_id: str, retry: int = 0) -> requests.Response:
    """
    Send a POST request to the specified endpoint with retries.

    Args:
        endpoint: The API endpoint to send the request to.
        payload: The data to send in the request body.
        job_id: The ID of the current job for logging.
        retry: Current retry attempt number.

    Returns:
        The response from the server.
    """
    response = session.post(
        url=f'{BASE_URI}/{endpoint}',
        json=payload,
        timeout=TIMEOUT
    )

    if response.status_code == 404 and retry < POST_RETRIES:
        retry += 1
        logger.warn(f'Received HTTP 404 from endpoint: {endpoint}, Retrying: {retry}', job_id)
        time.sleep(0.2)
        return send_post_request(endpoint, payload, job_id, retry)

    return response


def validate_input(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the input payload against the input schema.

    Args:
        job: The job dictionary containing the input to validate.

    Returns:
        The validated input or error dictionary.
    """
    return validate(job['input'], INPUT_SCHEMA)


def validate_api(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the API configuration against the API schema.

    Args:
        job: The job dictionary containing the API configuration to validate.

    Returns:
        The validated API configuration or error dictionary.
    """
    api = job['input']['api']
    api['endpoint'] = api['endpoint'].lstrip('/')

    return validate(api, API_SCHEMA)


def extract_scheduler(value: Union[str, int]) -> Tuple[Optional[str], str]:
    """
    Extract scheduler suffix from a sampler value if present.
    Preserves the original case of the value while checking for lowercase suffixes.

    Args:
        value: The sampler value to check for a scheduler suffix.

    Returns:
        A tuple containing the scheduler suffix (if found) and the cleaned value.
    """
    scheduler_suffixes = ['uniform', 'karras', 'exponential', 'polyexponential', 'sgm_uniform']
    value_str = str(value)
    value_lower = value_str.lower()

    for suffix in scheduler_suffixes:
        if value_lower.endswith(suffix):
            return suffix, value_str[:-(len(suffix))].rstrip()

    return None, value_str


def validate_payload(job: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """
    Validate the payload based on the endpoint and handle sampler/scheduler compatibility.

    Args:
        job: The job dictionary containing the payload to validate.

    Returns:
        A tuple containing the endpoint, method, and validated payload.
    """
    method = job['input']['api']['method']
    endpoint = job['input']['api']['endpoint']
    payload = job['input']['payload']
    validated_input = payload

    if endpoint in ['sdapi/v1/txt2img', 'sdapi/v1/img2img']:
        for field in ['sampler_index', 'sampler_name']:
            if field in payload:
                scheduler, cleaned_value = extract_scheduler(payload[field])
                if scheduler:
                    payload[field] = cleaned_value
                    payload['scheduler'] = scheduler

    if endpoint == 'v1/sync':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, SYNC_SCHEMA)
    elif endpoint == 'v1/download':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, DOWNLOAD_SCHEMA)
    elif endpoint == 'sdapi/v1/txt2img':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, TXT2IMG_SCHEMA)
    elif endpoint == 'sdapi/v1/img2img':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, IMG2IMG_SCHEMA)
    elif endpoint == 'sdapi/v1/interrogate' and method == 'POST':
        logger.info(f'Validating /{endpoint} payload', job['id'])
        validated_input = validate(payload, INTERROGATE_SCHEMA)

    return endpoint, job['input']['api']['method'], validated_input


def download(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download a file from a URL to a specified path.

    Args:
        job: The job dictionary containing the download configuration.

    Returns:
        A dictionary containing the download status and details.
    """
    source_url = job['input']['payload']['source_url']
    download_path = job['input']['payload']['download_path']
    process_id = os.getpid()
    temp_path = f'{download_path}.{process_id}'

    with requests.get(source_url, stream=True) as r:
        r.raise_for_status()
        with open(temp_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    os.rename(temp_path, download_path)
    logger.info(f'{source_url} successfully downloaded to {download_path}', job['id'])

    return {
        'msg': 'Download successful',
        'source_url': source_url,
        'download_path': download_path
    }


def sync(job: Dict[str, Any]) -> Dict[str, Union[int, List[str]]]:
    """
    Sync files from a Hugging Face repository to a local directory.

    Args:
        job: The job dictionary containing the sync configuration.

    Returns:
        A dictionary containing the sync status and details.
    """
    repo_id = job['input']['payload']['repo_id']
    sync_path = job['input']['payload']['sync_path']
    hf_token = job['input']['payload']['hf_token']

    api = HfApi()
    models = api.list_repo_files(
        repo_id=repo_id,
        token=hf_token
    )

    synced_count = 0
    synced_files = []

    for model in models:
        folder = os.path.dirname(model)
        dest_path = f'{sync_path}/{model}'

        if folder and not os.path.exists(dest_path):
            logger.info(f'Syncing {model} to {dest_path}', job['id'])

            uri = api.hf_hub_download(
                token=hf_token,
                repo_id=repo_id,
                filename=model,
                local_dir=sync_path,
                local_dir_use_symlinks=False
            )

            if uri:
                synced_count += 1
                synced_files.append(dest_path)

    return {
        'synced_count': synced_count,
        'synced_files': synced_files
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler function for processing RunPod jobs.

    Validates input, API configuration, and payload, then processes the request
    according to the specified endpoint and method.

    Args:
        job: The job dictionary containing all input data and configuration.

    Returns:
        A dictionary containing the job results or error information.
    """
    validated_input = validate_input(job)

    if 'errors' in validated_input:
        return {
            'error': '\n'.join(validated_input['errors'])
        }

    validated_api = validate_api(job)

    if 'errors' in validated_api:
        return {
            'error': '\n'.join(validated_api['errors'])
        }

    endpoint, method, validated_payload = validate_payload(job)

    if 'errors' in validated_payload:
        return {
            'error': '\n'.join(validated_payload['errors'])
        }

    if 'validated_input' in validated_payload:
        payload = validated_payload['validated_input']
    else:
        payload = validated_payload

    try:
        logger.info(f'Sending {method} request to: /{endpoint}', job['id'])

        if endpoint == 'v1/download':
            return download(job)
        elif endpoint == 'v1/sync':
            return sync(job)
        elif method == 'GET':
            response = send_get_request(endpoint)
        elif method == 'POST':
            response = send_post_request(endpoint, payload, job['id'])

        if response.status_code == 200:
            return response.json()

        resp_json = response.json()
        logger.error(f'HTTP Status code: {response.status_code}', job['id'])
        logger.error(f'Response: {resp_json}', job['id'])

        if 'error' in resp_json and 'errors' in resp_json:
            error = resp_json.get('error')
            errors = resp_json.get('errors')
            error_msg = f'{error}: {errors}'
        else:
            error_msg = f'A1111 status code: {response.status_code}'

        return {
            'error': error_msg,
            'output': resp_json,
            'refresh_worker': True
        }

    except Exception as e:
        logger.error(f'An exception was raised: {e}')
        return {
            'error': traceback.format_exc(),
            'refresh_worker': True
        }


if __name__ == '__main__':
    wait_for_service(f'{BASE_URI}/sdapi/v1/sd-models')
    logger.info('A1111 Stable Diffusion API is ready')
    logger.info('Starting RunPod Serverless...')
    runpod.serverless.start({
        'handler': handler
    })