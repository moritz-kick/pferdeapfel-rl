from typing import Any


def parse_toon(content: str) -> dict[str, Any]:
    """
    Simple parser for TOON format.
    Handles basic key-value pairs, lists, and nested objects with indentation.
    """
    lines = content.splitlines()
    root: dict[str, Any] = {}
    # Stack stores: (container, indentation_level)
    stack: list[tuple[Any, int]] = [(root, -1)]

    for line in lines:
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip())
        text = line.strip()

        # Adjust stack based on indentation
        while len(stack) > 1 and indent <= stack[-1][1]:
            stack.pop()

        current_container, _ = stack[-1]

        if ": " in text:
            key_part, value_part = text.split(": ", 1)
            key = key_part.split("[")[0]

            # Parse value
            value: Any = value_part
            if value_part == "null":
                value = None
            elif value_part == "true":
                value = True
            elif value_part == "false":
                value = False
            elif "," in value_part:
                try:
                    value = [int(x.strip()) for x in value_part.split(",")]
                except ValueError:
                    value = [x.strip() for x in value_part.split(",")]
            elif value_part.isdigit():
                value = int(value_part)

            if isinstance(current_container, dict):
                current_container[key] = value
            elif isinstance(current_container, list):
                # If list of objects, we might need to start a new object
                if not current_container or key in current_container[-1]:
                    current_container.append({})
                current_container[-1][key] = value

        elif text.endswith(":"):
            key_part = text[:-1]
            key = key_part.split("[")[0]
            is_list = "[" in key_part

            new_container: list[Any] | dict[str, Any] = [] if is_list else {}

            if isinstance(current_container, dict):
                current_container[key] = new_container
            elif isinstance(current_container, list):
                if not current_container or key in current_container[-1]:
                    current_container.append({})
                current_container[-1][key] = new_container

            stack.append((new_container, indent))

    return root
