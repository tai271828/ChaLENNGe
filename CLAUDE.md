# Runtime Management
## Python Runtime
- use system wide uv to manage python virtual environment .venv

## Agentic GenAI Runtime
- Claude
- Something else. Use symbolic links to share default context and skills

## Reply style

End every reply that completes a task with a `## Summary` section containing,
in this order:

1. **Summary line** — one line, 50 words or fewer.
2. **Key items** — a bullet list; each bullet 25 words or fewer.
3. **Next step** — one suggested next step, 25 words or fewer.

# Documentation
- Use consise but meaningful comments. The goal is to minimize cognitive load when reading the code, while still providing enough information to understand the purpose and functionality of each part of the code.
- Use docstrings for functions and classes to explain their purpose, parameters, and return values.
- Remember to update the markdown documentation in the repository to reflect any changes made to the codebase

# Formatting
- Always use black for code formatting to ensure consistency across the codebase.
