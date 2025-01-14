import requests

conversations = [[{"role": "user", "content": "你是谁"}], [{"role": "user", "content": "小炒肉怎么做？"}]]
response = requests.post("http://127.0.0.1:8000/", json=conversations)
print(response.json())
