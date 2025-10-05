from httpx import delete
import requests


class QdrantClient:
    def __init__(self, url: str, api_key: str = None):
        self.url = url
        self.api_key = api_key

    # Upsert points into a collection
    def upsert_points(self, collection_name: str, points: list):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "points": points
        }

        response = requests.put(
            f"{self.url}/collections/{collection_name}/points",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    
    # Create a collection if it does not exist
    def create_collection(self, collection_name: str, vector_size: int = 768, distance: str = "Cosine"):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "vectors": {
                "size": vector_size,
                "distance": distance
            }
        }

        response = requests.put(
            f"{self.url}/collections/{collection_name}",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    
    # Check if a collection exists
    def collection_exists(self, collection_name: str) -> bool:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.get(
            f"{self.url}/collections/{collection_name}",
            headers=headers
        )
        return response.status_code == 200
    
    # Delete a collection
    def delete_collection(self, collection_name: str):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = delete(
            f"{self.url}/collections/{collection_name}",
            headers=headers
        )
        response.raise_for_status()
        return response.json()