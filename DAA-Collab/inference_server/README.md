# Remote Inference Server

This folder contains the server and client for offloading GPU-heavy model inference to a remote machine.

## Architecture

- **Server** (`server.py`): Runs on GPU machine, hosts embedding and LLM models
- **Client** (`client.py`): Lightweight wrapper to call server, can run on any machine

## Setup

### Server (GPU Machine)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
python server.py --host 0.0.0.0 --port 8000
```

The server will:
- Load models on startup (takes ~30 seconds)
- Keep models in GPU memory
- Listen for inference requests

### Client (Laptop/Any Machine)

1. Install only client dependencies:
```bash
pip install requests numpy
```

2. Test connection:
```bash
python client.py http://your-gpu-server-ip:8000
```

## API Endpoints

### POST /embed
Generate embeddings for text

**Request:**
```json
{
  "texts": ["query text", "intent 1", "intent 2"]
}
```

**Response:**
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
}
```

### POST /extract-intents
Extract unknown intents from query

**Request:**
```json
{
  "query": "user query text",
  "known_intents": ["intent1", "intent2"],
  "known_classes": ["class1", "class2"]
}
```

**Response:**
```json
{
  "unknown_intents": ["new_intent1", "new_intent2"]
}
```

### GET /health
Check server status

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "embedding_model_loaded": true,
  "llm_model_loaded": true
}
```

## Integration with Existing Code

To use the server in your existing intent classification:

```python
from inference_server.client import InferenceClient
import os

# Initialize client
SERVER_URL = os.getenv("INFERENCE_SERVER_URL", "http://localhost:8000")
client = InferenceClient(SERVER_URL)

# Replace local embedding calls
# OLD: embeddings = embedding_model.encode(texts)
# NEW: 
embeddings = client.get_embeddings(texts)

# Replace local LLM extraction calls
# OLD: unknown = extract_unknown_intents_llm(query, known, classes)
# NEW:
unknown = client.extract_unknown_intents(query, known, classes)
```

## Performance

**Server (GPU):**
- Memory: ~7.9GB GPU (8-bit embedding + 4-bit LLM)
- Startup: ~30 seconds
- Per request: <2 seconds

**Client:**
- Memory: ~50MB (no models)
- Network: 2-3 API calls per query
- Latency: +100-500ms vs local (depends on network)

## Firewall/Network

Make sure port 8000 is open on the server:
```bash
# Linux
sudo ufw allow 8000

# Windows
netsh advfirewall firewall add rule name="Inference Server" dir=in action=allow protocol=TCP localport=8000
```

## Running on Different Machines

**Same computer** (testing):
```bash
# Terminal 1 - Start server
python server.py

# Terminal 2 - Test client
python client.py http://localhost:8000
```

**Different computers** (production):
```bash
# GPU machine (e.g., 192.168.1.100)
python server.py --host 0.0.0.0 --port 8000

# Laptop
export INFERENCE_SERVER_URL=http://192.168.1.100:8000
python client.py $INFERENCE_SERVER_URL
```

## Troubleshooting

**Server won't start:**
- Check CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify GPU memory: `nvidia-smi`
- Check port not in use: `netstat -an | findstr 8000`

**Client can't connect:**
- Verify server is running: `curl http://server-ip:8000/health`
- Check firewall allows port 8000
- Ping server: `ping server-ip`

**Out of memory:**
- Server uses ~8GB GPU memory
- If insufficient, models will fail to load
- Check GPU memory: `nvidia-smi`
