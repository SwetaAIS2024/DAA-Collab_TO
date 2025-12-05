# # Node that updates instructions


# def update_instructions(state: State, store: BaseStore):
#     namespace = ("instructions",)
#     instructions = store.search(namespace)[0]
#     # Memory logic
#     prompt = prompt_template.format(instructions=instructions.value["instructions"], conversation=state["messages"])
#     output = llm.invoke(prompt)
#     new_instructions = output['new_instructions']
#     store.put(("agent_instructions",), "agent_a", {"instructions": new_instructions})
#     ...
