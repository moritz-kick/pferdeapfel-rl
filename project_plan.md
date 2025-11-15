
# project_plan.md

## Project Structure Overview

This project implements the Pferdeäpfel game with RL training using Stable Baselines3. The structure is modular, with separation for game logic, players, GUI, environments, training, evaluation, and data/logs. We'll use Python 3, uv for dependency management, and pyproject.toml for setup. Add standard files for best practices (e.g., LICENSE, README, tests). Since Cursor (AI coding tool) is used, include a .cursorignore if needed (similar to .gitignore for AI tools), and ensure code is clean/modular for easy editing.

### Overall Directory Structure
```
pferdeapfel-rl/
├── .gitignore
├── README.md           # Game rules and overview
├── project_plan.md     # This file
├── pyproject.toml      # Dependencies, scripts, metadata
├── data/
│   ├── logs/
│   │   ├── game/       # JSON logs of games
│   │   ├── player/     # JSON logs of players
│   │   └── rl_player/  # JSON logs of RL I/O checks
│   └── models/         # Saved RL models (e.g., .zip from SB3)
├── src/
│   ├── __init__.py
│   ├── game/
│   │   ├── __init__.py
│   │   ├── board.py    # Board state, moves, validation
│   │   └── rules.py    # Core rules, win conditions
│   ├── players/
│   │   ├── __init__.py
│   │   ├── base.py     # Abstract player class
│   │   ├── human.py
│   │   ├── random.py
│   │   └── rl/
│   │       ├── __init__.py
│   │       ├── dqn_rl.py     # DQN RL player
│   │       └── ppo_rl.py     # PPO RL player
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── main.py         # GUI entry point (using PySide6)
│   │   ├── config.json     # Default GUI settings (players, logs, etc.)
│   │   └── dashboard.py    # Visualization for benchmarks/evals
│   ├── env/
│   │   ├── __init__.py
│   │   └── pferdeapfel_env.py  # Gym-compatible env for SB3
│   ├── training/
│   │   ├── __init__.py
│   │   ├── config.yaml      # Configuration for training
│   │   ├── dqn_train.py     # DQN training script
│   │   ├── ppo_train.py     # PPO training script
│   │   └── optuna_tune.py   # Optuna tuning script
│   └── evaluation/
│       ├── __init__.py
│       ├── eval_script.py  # Full pairwise evaluation, skip existing
│       └── results.csv     # Eval outputs
├── tests/
│   ├── __init__.py
│   ├── test_legal_moves.py     # Test legal moves
│   ├── test_win_conditions.py  # Test win conditions
│   └── test_rl_io.py           # Test RL player I/O
└── scripts/
    ├── simulate_random.py  # Random vs random simulation
    └── run_gui.py          # Entry script for GUI
```

### Milestones

#### Milestone 0: Repo Setup
1. Create Git repo (e.g., `git init pferdeapfel-rl`).
2. Install uv: `pip install uv` (or globally).
3. Initialize project: `uv init` to create pyproject.toml.
4. Edit pyproject.toml: Add dependencies like `stable-baselines3`, `gymnasium` (for env), `torch` (backend), `optuna`, `tensorboard`, `PySide6` for GUI, `polars` for data handling, `matplotlib` for dashboard, `ruff` for linting, `mypy` for type checking.
   - Example: [dependencies] section with versions.
   - Add [tool.uv] for config if needed.
   - Include scripts: e.g., [tool.uv.scripts] gui = "python scripts/run_gui.py"
5. Create .gitignore: Standard Python (ignore __pycache__, .venv, *.pyc, data/logs/*).
6. Add README.md & project_plan.md files.
7. Create directories as per structure.
8. Commit initial setup.
9. Set up virtual env: `uv venv` and activate.
10. Install deps: `uv sync`.
11. Add .pre-commit-config.yaml file:
```yaml
repos:
  - repo: local
    hooks:
      - id: ruff
        name: ruff
        entry: uv run ruff check
        language: system
        types: [python]
        args: ["--fix", "--exit-non-zero-on-fix"]

      - id: mypy
        name: mypy
        entry: uv run mypy
        language: system
        types: [python]

```

#### Milestone 1: Simple Game Implementation
1. Implement core game in src/game/: board.py for state (numpy array for board), rules.py for moves/validation/win checks.
2. Create player classes in src/players/: base.py with abstract methods (e.g., get_move(state)), human.py (input-based), random.py (choose random legal move).
3. Build GUI in src/gui/: Use PySide6 for 8x8 grid display.
   - Show current state: Board with horses, blocked squares, apple counts, turn indicator.
   - Display legal moves: Highlight possible knight jumps.
   - Player selection: Dropdown for White/Black (human, random, or later RL).
   - Starting player: Randomly choose at start.
   - Logging: Toggle button; if on, save game states/moves as JSON in data/logs/game/ (e.g., game_001.json).
   - Redo last move: Undo button to revert state (keep a move history stack).
   - Display Winner: Show winner when game ends.
   - Restart button for new game.
4. Add simulation script: scripts/simulate_random.py to run random vs random, output to console or log.
5. Basic tests in tests/: Ensure legal moves, win conditions work.
6. Run/test: Play human vs random, verify features.

#### Milestone 2: RL Environment and Basic Training
1. Create Gym env in src/env/pferdeapfel_env.py: Inherit from gymnasium.Env.
   - Observation space: Flattened state (board flat + positions + turn + brown_left + golden_flag).
   - Action space: Discrete or MultiDiscrete (move index + optional placement coords or skip).
   - step(): Handle move + add apple + optional place, check validity, return obs/reward/done.
   - Reward: win/loss, illegal move (game over, penalty), legal move (small reward).
   - Optional: Support rendering for GUI integration.
2. First RL player in src/players/rl/dqn_rl.py: Wrap SB3 model, load/save models.
3. I/O check script: tests/test_rl_io.py – Input sample states, log actions/rewards to data/logs/rl_player/.
4. Training in src/training/:
   - config.yaml: Params like learning_rate, n_steps, batch_size.
   - ppo_train.py: Load config, create env, train PPO, save model to data/models/, use TensorBoard for logging (e.g., callbacks).
5. Integrate RL to GUI: Scan src/players/rl/ for subclasses, add to dropdown; load defaults from gui/config.json (e.g., {"white": "human", "black": "random", "logs": true}).
6. Evaluation: In src/evaluation/eval_script.py – Run N games RL vs random, compute win rate, save to CSV.
7. Dashboard: src/gui/dashboard.py – Use matplotlib to plot win rates, viewable in GUI or separate.
8. (Added) Tests for env (valid spaces, steps).
9. Update README.md

#### Milestone 3: Advanced Training and Evaluation
1. Optuna integration: optuna_tune.py – Define objective func to train/tune hypers, run studies.
2. Advanced evaluation, update existing: src/evaluation/eval_script.py
   - Scan src/players/ (exclude human) and data/models/ for available players/models.
   - Load existing results.csv with columns: player1, player2, wins1, wins2, games_played.
   - For each pair (including self-play if useful), if not fully played <1000 games, run simulations, update file.
   - Skip human; support parallel runs.
   - Options: N games per pair, randomroles (White/Black).
3. Integrate eval results to dashboard for visualization (heatmaps of win rates).
4. Add best move visualization to GUI
5. Update README.md