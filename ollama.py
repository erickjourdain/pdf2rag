import requests

class OllamaClient:
    def __init__(self, url: str = "http://localhost:11434/api"):
        self.url = url

    def embeddings(text: str, url: str, model: str = "nomic-embed-text") -> list:
        payload = {
            "model": model,
            "prompt": text
        }
        resp = requests.post(url + "/embeddings", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("embedding", [])