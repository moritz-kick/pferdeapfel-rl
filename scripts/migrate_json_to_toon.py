import json
import toon_python
from pathlib import Path


def migrate_logs():
    log_dirs = [Path("debug_logs"), Path("data/logs/game")]

    for log_dir in log_dirs:
        if not log_dir.exists():
            print(f"No directory found at {log_dir}")
            continue

        json_files = list(log_dir.glob("*.json"))
        if not json_files:
            print(f"No JSON logs found in {log_dir} to migrate.")
            continue

        print(f"Found {len(json_files)} JSON logs in {log_dir} to migrate.")

        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                toon_content = toon_python.encode(data)
                toon_file = json_file.with_suffix(".toon")

                with open(toon_file, "w") as f:
                    f.write(toon_content)

                print(f"Migrated {json_file.name} -> {toon_file.name}")

                # Delete original file
                json_file.unlink()

            except Exception as e:
                print(f"Failed to migrate {json_file.name}: {e}")


if __name__ == "__main__":
    migrate_logs()
