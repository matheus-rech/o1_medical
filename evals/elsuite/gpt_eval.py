from openai import AzureOpenAI
import re
from loguru import logger
from tenacity import (
    retry,
    wait_random_exponential,
)  # for exponential backoff
import random
System_prompt = '''You are a senior medical expert. Please evaluate the quality of the medical text material provided by medical interns based on the expert medical text material as a reference answer. The quality is divided into five levels:

5. The assistant result completely matches the reference.
4. The assistant result is generally consistent with the reference, with only a small part of omissions or errors.
3. The assistant result partially matches the reference, but there are some omissions and errors.
2. The assistant result is mostly inconsistent with the reference, with many omissions and errors.
1. The assistant result is completely inconsistent with the reference.

'''

Eval_prompt = '''
{prompt}

[Assistant Result]
{result}

[Reference]
{reference}

Please note:
(1) Focus on the factual content of the medical answers, without concern for style, grammar, punctuation, and non-medical content.
(2) Your response should be in the format.
Rating: (int)'''


endpoint_key_gpt4_turbo = {
    "ue2": ["https://vlaa-openai-eastus2.openai.azure.com/", "gpt-4-1106-preview-nofilter", "23196628a28f4badb0f71a32e8406321"],
    "uw": ["https://openai-vlaa-westus.openai.azure.com", "gpt-4-1106-preview-nofilter", "c235fce767564930b9571e4840943c75"],
    "us": ["https://openai-vlaa-uksouth.openai.azure.com", "gpt-4-1106-preview-nofilter", "632a45f0552449d5bf2e9c3843eafcdc"],
    "sc": ["https://openai-vlaa-swedencentral.openai.azure.com", "gpt-4-1106-preview-nofilter", "4024794d0d9c457f9b7407e1753dc93b"],
    "ce": ["https://openai-vlaa-canadaeast.openai.azure.com", "gpt-4-1106-preview-nofilter", "31fe73fad0f54035bd71d8a926fccefa"],
    "fc": ["https://openai-vlaa-francecentral.openai.azure.com", "gpt-4-1106-preview-nofilter", "4bf9f9de2a29406dac270d5e53b92dbc"],
}

def create_client(key_list):
    client = AzureOpenAI(
        azure_endpoint=key_list[0],
        api_key=key_list[2],
        api_version="2023-12-01-preview"
    )
    return client, key_list[1]

def _extract_score_from_reka_output(evaluator_response: str):
    """
    Extract the score from the evaluator response. Refer to the official Vibe-Eval implementation:
    https://github.com/reka-ai/reka-vibe-eval/blob/3852d4712da172a7b85dddeffc4f9c3482a6f4c9/evaluate.py#L159-#L164
    """
    re_match = re.search(r"Rating:\s*([1-5])", evaluator_response)
    if re_match is None:
        return None
    return {"quality": int(re_match.group(1))}
def _log_when_fail(retry_state):
    logger.info(
        "Request failed. Current retry attempts:{}. Sleep for {:.2f}. Exception: {}".format(
            retry_state.attempt_number, retry_state.idle_for, repr(retry_state.outcome.exception())
        )
    )
@retry(
    wait=wait_random_exponential(min=1, max=60),
    before_sleep=_log_when_fail
)
def get_score(prompt: str, result: str, reference: str, endpoint: str = "ue2") -> int:
    """
    Get the score from the evaluator response.
    """
    endpoint = random.choice(list(endpoint_key_gpt4_turbo.keys()))
    client, client_name = create_client(endpoint_key_gpt4_turbo[endpoint])

    response = client.chat.completions.create(
        model=client_name,
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": System_prompt},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": Eval_prompt.format(prompt=prompt, result=result, reference=reference)},
                ]
            }
        ],
    )
    return _extract_score_from_reka_output(response.choices[0].message.content)





