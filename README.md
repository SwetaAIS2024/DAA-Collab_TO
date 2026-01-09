### Test steps :

1. Activate the virtual environment (one time)
    ```
    .\d_venv\Scripts\Activate.ps1
    ```

2. Install requirements (one time)
    ```
    pip install -r .\DAA-Collab\requirements.txt
    ```

3. Generate JWT secret (one time)
    ```
    d_venv\Scripts\chainlit.exe create-secret
    ```
    Copy the generated secret to `DAA-Collab\.env`:
    ```
    CHAINLIT_AUTH_SECRET="<your-generated-secret>"
    ```

4. Build intent centroids
    ```
    python .\DAA-Collab\utils\ML\ml_based_intent_classification\build_centroids_script.py
    ```

5. Start Chainlit server
    ```
    chainlit run .\DAA-Collab\app_chainlit.py
    ```

6. Open http://localhost:8000/

### Testing Planner + CodeGen Nodes:

To test the new planner and code generator nodes independently:

```
python .\DAA-Collab\test_planner_codegen.py
```

This demonstrates:
- Intent → Capability mapping
- Missing tool detection
- CodeGen LLM input JSON generation
- Tool package generation (simulated)
- Sandbox validation (simulated)
- Tool registration (simulated)

Output files saved to: `DAA-Collab/data/codegen_requests/`

### Features:
- Intent classification with confidence scores
- User feedback mechanism (Correct/Incorrect buttons)
- Chat history persistence (survives page refresh)
- Custom unknown intent tagging
- All predictions sorted by score
- **NEW:** Planner node (micro-planner MVP)
- **NEW:** CodeGen node (tool generation pipeline)

### Response example :
![alt text](example_query_response.png)

### User feedback feature :

If the intents given by the model is not correct, user can modify or add the new intent tags.

#### Example :
1. example query :
![alt text](image-1.png)

2. user feedback :
![alt text](image-3.png)

$env:CHAINLIT_AUTH_SECRET="2SDaId82NtdRiI*OL.X*,MV7gZuu7t3vVjm*?z5ZMvD650r4k.N%QYu6:%oXXmrc"; chainlit run .\DAA-Collab\app_chainlit.py