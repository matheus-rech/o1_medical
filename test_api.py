from openai import AzureOpenAI  # Requires openai>=1.0.0

endpoint_key_gpt4_turbo = {
    # "ue2": ["https://vlaa-openai-eastus2.openai.azure.com/", "gpt-4-1106-preview-nofilter", "23196628a28f4badb0f71a32e8406321"],
    "uw": ["https://openai-vlaa-westus.openai.azure.com", "gpt-4-1106-preview-nofilter", "c235fce767564930b9571e4840943c75"],
    # "us": ["https://openai-vlaa-uksouth.openai.azure.com", "gpt-4-1106-preview-nofilter", "632a45f0552449d5bf2e9c3843eafcdc"],
    # "sc": ["https://openai-vlaa-swedencentral.openai.azure.com", "gpt-4-1106-preview-nofilter", "4024794d0d9c457f9b7407e1753dc93b"],
    # "ce": ["https://openai-vlaa-canadaeast.openai.azure.com", "gpt-4-1106-preview-nofilter", "31fe73fad0f54035bd71d8a926fccefa"],
    # "fc": ["https://openai-vlaa-francecentral.openai.azure.com", "gpt-4-1106-preview-nofilter", "4bf9f9de2a29406dac270d5e53b92dbc"],
}

for region, (endpoint, deployment_name, api_key) in endpoint_key_gpt4_turbo.items():
    # Initialize the AzureOpenAI client
    client = AzureOpenAI(
        azure_endpoint=endpoint.rstrip('/'),
        api_key=api_key,
        api_version="2023-12-01-preview"
    )

    # Prepare the messages for the chat completion
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."}
            ]
        },
    ]

    try:
        # Make the API call to create a chat completion
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages
        )
        print(f"Region {region}: API call successful.")
        print(response)
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Region {region}: API call failed with error: {e}")
