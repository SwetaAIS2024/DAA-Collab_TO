"""
Client wrapper for remote inference server
Can be imported and used in place of local model loading
"""

import requests
from typing import List, Tuple, Dict
import numpy as np


class InferenceClient:
    """Client for communicating with the inference server"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        Initialize client
        
        Args:
            server_url: URL of the inference server (e.g., "http://192.168.1.100:8000")
        """
        self.server_url = server_url.rstrip('/')
        self._check_connection()
    
    def _check_connection(self):
        """Check if server is reachable"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            response.raise_for_status()
            health = response.json()
            if not health.get("embedding_model_loaded") or not health.get("llm_model_loaded"):
                raise ConnectionError("Server models not fully loaded")
            print(f"Connected to inference server: {self.server_url}")
            print(f"Device: {health.get('device', 'unknown')}")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to inference server: {e}")
    
    def get_embeddings(self, texts: List[str], timeout: int = 30) -> np.ndarray:
        """
        Get embeddings for texts from remote server
        
        Args:
            texts: List of text strings to embed
            timeout: Request timeout in seconds
            
        Returns:
            numpy array of embeddings, shape (len(texts), embedding_dim)
        """
        try:
            response = requests.post(
                f"{self.server_url}/embed",
                json={"texts": texts},
                timeout=timeout
            )
            response.raise_for_status()
            embeddings = response.json()["embeddings"]
            return np.array(embeddings)
        except Exception as e:
            raise RuntimeError(f"Embedding request failed: {e}")
    
    def extract_unknown_intents(
        self,
        query: str,
        known_intents: List[str],
        known_classes: List[str],
        timeout: int = 60
    ) -> List[str]:
        """
        Extract unknown intents from query using remote LLM
        
        Args:
            query: User query text
            known_intents: List of already identified intents
            known_classes: List of all known intent classes
            timeout: Request timeout in seconds
            
        Returns:
            List of unknown intent names
        """
        try:
            response = requests.post(
                f"{self.server_url}/extract-intents",
                json={
                    "query": query,
                    "known_intents": known_intents,
                    "known_classes": known_classes
                },
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()["unknown_intents"]
        except Exception as e:
            raise RuntimeError(f"Intent extraction request failed: {e}")


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage: python client.py <server_url>")
        print("Example: python client.py http://192.168.1.100:8000\n")
        sys.exit(1)
    
    server_url = sys.argv[1]
    
    # Test connection
    print(f"\nTesting connection to {server_url}...")
    client = InferenceClient(server_url)
    
    # Test embedding
    print("\nTesting embedding generation...")
    texts = ["traffic congestion", "road closure", "accident report"]
    embeddings = client.get_embeddings(texts)
    print(f"Generated embeddings: shape={embeddings.shape}")
    print(f"Sample embedding (first 5 dims): {embeddings[0][:5]}")
    
    # Test intent extraction
    print("\nTesting unknown intent extraction...")
    query = "How is the main highway 4 waterworks impacting the traffic?"
    known_intents = ["traffic_impact", "traffic_congestion"]
    known_classes = ["traffic_impact", "traffic_congestion", "visualization", "email_notification"]
    
    unknown = client.extract_unknown_intents(query, known_intents, known_classes)
    print(f"Unknown intents found: {unknown}")
    
    print("\n✓ All tests passed!")
