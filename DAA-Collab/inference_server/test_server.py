"""
Quick test script to verify server is working correctly
"""

from client import InferenceClient
import sys

def test_server(server_url: str):
    """Run comprehensive tests on the inference server"""
    
    print(f"\n{'='*60}")
    print(f"Testing Inference Server: {server_url}")
    print(f"{'='*60}\n")
    
    # Test 1: Connection
    print("Test 1: Server Connection")
    print("-" * 40)
    try:
        client = InferenceClient(server_url)
        print("✓ Server is reachable and models loaded\n")
    except Exception as e:
        print(f"✗ Connection failed: {e}\n")
        return False
    
    # Test 2: Embedding generation
    print("Test 2: Embedding Generation")
    print("-" * 40)
    try:
        texts = [
            "traffic congestion on highway",
            "road closure notification",
            "accident report details"
        ]
        embeddings = client.get_embeddings(texts)
        print(f"Input texts: {len(texts)}")
        print(f"Output shape: {embeddings.shape}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"Sample values: {embeddings[0][:3]}")
        print("✓ Embeddings generated successfully\n")
    except Exception as e:
        print(f"✗ Embedding test failed: {e}\n")
        return False
    
    # Test 3: Unknown intent extraction
    print("Test 3: Unknown Intent Extraction")
    print("-" * 40)
    try:
        query = "How is the main highway 4 waterworks impacting the traffic? Give full analysis with graphs."
        known_intents = ["traffic_impact", "traffic_congestion"]
        known_classes = [
            "traffic_impact", "traffic_congestion", "visualization",
            "email_notification", "meta_attributes", "causal_analysis"
        ]
        
        print(f"Query: {query}")
        print(f"Known intents: {known_intents}")
        print(f"Known classes: {len(known_classes)}")
        
        unknown = client.extract_unknown_intents(query, known_intents, known_classes)
        
        print(f"\nExtracted unknown intents: {unknown}")
        print(f"Count: {len(unknown)}")
        print("✓ Intent extraction completed\n")
    except Exception as e:
        print(f"✗ Intent extraction failed: {e}\n")
        return False
    
    # Test 4: Multiple queries (stress test)
    print("Test 4: Batch Processing")
    print("-" * 40)
    try:
        test_queries = [
            "Show me traffic patterns",
            "Send email alerts for congestion",
            "Analyze incident impact on routes"
        ]
        
        for i, query in enumerate(test_queries, 1):
            emb = client.get_embeddings([query])
            print(f"  Query {i}: ✓ (embedding shape: {emb.shape})")
        
        print("✓ Batch processing successful\n")
    except Exception as e:
        print(f"✗ Batch test failed: {e}\n")
        return False
    
    print("="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
    print("\nServer is ready for production use!")
    print(f"Client code can connect to: {server_url}\n")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        server_url = "http://localhost:8000"
        print(f"No URL provided, using default: {server_url}")
    else:
        server_url = sys.argv[1]
    
    success = test_server(server_url)
    sys.exit(0 if success else 1)
