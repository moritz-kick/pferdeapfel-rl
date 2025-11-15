"""Entry script for running the GUI."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gui.main import main

if __name__ == "__main__":
    main()
