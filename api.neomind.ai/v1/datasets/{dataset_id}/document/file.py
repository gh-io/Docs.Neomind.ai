import requests

url = "https://api.neomind.ai/v1/datasets/{dataset_id}/document/create-by-file"

files = { "file": ("example-file", open("example-file", "rb")) }
payload = { "data": "{\"indexing_technique\":\"high_quality\",\"process_rule\":{\"mode\":\"custom\", \"rules\": { \"segmentation\": {\"separator\":\"###\", \"max_tokens\":500}}}}" }
headers = {"Authorization": "Bearer <token>"}

response = requests.post(url, data=payload, files=files, headers=headers)

print(response.text)
