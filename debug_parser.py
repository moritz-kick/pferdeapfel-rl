from typing import Any


def parse_toon_improved(content: str) -> dict[str, Any]:
    lines = content.splitlines()
    root: dict[str, Any] = {}
    # Stack stores: (container, indentation_level)
    stack: list[tuple[Any, int]] = [(root, -1)]

    print(f"DEBUG: Starting parse. Lines: {len(lines)}")

    for i, line in enumerate(lines):
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip())
        text = line.strip()

        print(f"DEBUG: Line {i}: '{text}' (indent {indent})")

        # Adjust stack based on indentation
        while len(stack) > 1 and indent <= stack[-1][1]:
            popped, level = stack.pop()
            print(f"DEBUG: Popped stack level {level}. Stack size: {len(stack)}")

        current_container, current_level = stack[-1]
        print(f"DEBUG: Current container type: {type(current_container)}, level: {current_level}")

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

            print(f"DEBUG: Key: {key}, Value: {value}")

            if isinstance(current_container, dict):
                current_container[key] = value
            elif isinstance(current_container, list):
                # If list of objects, we might need to start a new object
                if not current_container or key in current_container[-1]:
                    print("DEBUG: Starting new object in list")
                    current_container.append({})
                current_container[-1][key] = value

        elif text.endswith(":"):
            key_part = text[:-1]
            key = key_part.split("[")[0]
            is_list = "[" in key_part

            print(f"DEBUG: New container key: {key}, is_list: {is_list}")

            new_container = [] if is_list else {}

            if isinstance(current_container, dict):
                current_container[key] = new_container
            elif isinstance(current_container, list):
                if not current_container or key in current_container[-1]:
                    current_container.append({})
                current_container[-1][key] = new_container

            stack.append((new_container, indent))
            print(f"DEBUG: Pushed to stack. Level {indent}")

    return root


content = """white_player: White
black_player: Black
starting_player: white
winner: null
moves[17]:
  turn: black
  move_to[2]: 5,6
  extra_apple: null
"""

data = parse_toon_improved(content)
print("Result:", data)
