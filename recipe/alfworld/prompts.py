ALFWORLD_SYSTEM_PROMPT = (
    "You are acting in ALFWorld TextWorld. "
    "Output exactly one executable text command each turn. "
    "Do not explain. Do not answer in free-form text."
)


ALFWORLD_USER_PROMPT = """### Current Observation
{observation}

### History Actions
{history_actions}

### Instructions
- Use exactly one command through the `env_step` tool.
- Follow ALFWorld TextWorld command style such as `go to dresser 1`, `take mug 1 from cabinet 3`, `use desklamp 1`.
- Use the official observation text as the source of truth.
- Do not output explanations or a final natural-language answer.
"""


EXEC_ACTION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "env_step",
        "description": (
            "Execute one ALFWorld TextWorld command and return the next official observation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "A single ALFWorld TextWorld command such as "
                        "`go to dresser 1`, `open cabinet 3`, `take mug 1 from cabinet 3`, `use desklamp 1`."
                    ),
                }
            },
            "required": ["command"],
        },
    },
}


ALFWORLD_TOOL_SCHEMAS = [EXEC_ACTION_TOOL_SCHEMA]
