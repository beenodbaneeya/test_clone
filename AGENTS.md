# Project Guidelines

- Never modify files without explicit user approval.
- Use Python type hints for new code.
- Do not run anything to verify it as everything we have is local. Just change the files and I will copy and test it in the cluster
- Do not introduce dependencies without approval.
- Explain concept of deep learning whereever required



# Code generation guidelines
Code Quality & Architecture Guidelines

When writing or refactoring Python code, strictly follow these principles:

1. Keep It Lean & Functional: Write clean, minimal, production-ready code. Do not add over-engineered wrappers, unnecessary try/except blocks around simple logic, or boilerplate that adds line count without real value.

2. Avoid Output Clutter: Do not add manual warning suppression blocks, verbose custom loggers, or subprocess calls (e.g., calling shell commands like nvidia-smi or system tools) when standard Python or library APIs exist.

3. Native Standard Libraries First: Prefer built-in language features and native framework utilities over calling external process tools or reinventing functionality provided by established libraries.

4. Self-Contained & Complete: Ensure code includes all necessary operational steps (e.g., input checking, resource validation, parameter setup) to execute cleanly end-to-end without removing critical runtime logic.

5. Readability & Modular Structure:

- Organize scripts into clear, single-responsibility functions.
- Include standard CLI argument parsing (argparse) with sensible defaults where applicable.
- Use standard Python type hints and concise, high-value comments to explain why something is done, not what a basic line of code does.