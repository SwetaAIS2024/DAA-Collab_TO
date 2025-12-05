# DAA-Collab


DAA-Collab/
├── data
│   ├── dataset_from_DB_Agent
│   ├── MCP_tool_code_generation_dataset
│   └── user_prompt_dataset
├── func
│   ├── graph
|   │   └── graph_build.py
│   ├── memory
|   │   └── 
│   ├── nodes
│   │   ├── execute_node.py
│   │   ├── intent_classifer_node.py
│   │   ├── mcp_tool_generator_node.py
│   │   ├── planner_node.py
│   │   ├── tool_registration_node.py
|   │   └── update_instruction_node.py
│   ├── runner
|   │   └── run.py
│   └── state
|   │   └── agent_state.py
├── utils
│   ├── mcp_tools_registry
│   │   ├── generated_tools
│   │   ├── static_tools
|   │   └── smart_mcp_tool_lookup.py
│   ├── ML
│   │   ├── intent_extraction_classification
│   │   |   ├── intent_classification.py
│   │   |   ├── intent_extraction.py
│   │   |   ├── intent_keywords_static.py
│   │   |   ├── intent_mapping.py
|   │   |   └── ml_intent_classification.py
│   │   ├── mcp_tool_generation
|   │   |   └── mcp_tool_generation.py
|   │   └── common.py
│   └── common.py

├── main.py
├── README.md
├── .env # environment variables
├── requirements.txt # package dependencies
└── langgraph.json # configuration file for LangGraph
