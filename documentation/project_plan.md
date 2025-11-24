
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
│   │       ├── greedy.py     # Mobility-based greedy player for debugging
│   │       └── ppo_self_play.py  # PPO RL player (single policy self-play)
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── main.py         # GUI entry point (using PySide6)
│   │   ├── config.json     # Default GUI settings (players, logs, etc.)
│   │   └── dashboard.py    # Visualization for benchmarks/evals
│   ├── env/
│   │   ├── __init__.py
│   │   └── knight_self_play_env.py  # Gymnasium env with single-policy perspective swap
│   ├── training/
│   │   ├── __init__.py
│   │   ├── config.yaml      # Configuration for training
│   │   ├── ppo_self_play_train.py  # PPO training entry point
│   │   └── optuna_tune.py          # Optional hyper-parameter tuning
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
10. Install deps: `uv sync --all-groups`.
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

#### Milestone 1: Core Game Baseline + Greedy Debug Player
1. Implement core game in `src/game/` (`board.py`, `rules.py`) with validated knight moves and win conditions for mode 2 first.
2. Build player scaffolding in `src/players/`:
   - `base.py` abstract API, `human.py`, `random.py`, and new `greedy.py` that maximizes next-move mobility (used later for smoke tests).
3. Keep GUI lightweight for now: minimal PySide6 prototype to visualize moves and allow human vs random/greedy for debugging (full dashboard delayed).
   - Show current state: Board with horses, blocked squares, apple counts, turn indicator.
   - Display legal moves: Highlight possible knight jumps.
   - Player selection: Dropdown for White/Black (human, random, or later RL).
   - Starting player: Randomly choose at start.
   - Logging: Toggle button; if on, save game states/moves as JSON in data/logs/game/ (e.g., game_001.json).
   - Redo last move: Undo button to revert state (keep a move history stack).
   - Display Winner: Show winner when game ends.
   - Restart button for new game.
4. `scripts/simulate_random.py` now supports `{random, greedy}` combos for regression runs and writes JSON summaries.
5. Basic tests in `tests/`: legal moves, win conditions, greedy-move determinism.
6. Smoke-test CLI sims (human optional) to ensure deterministic seeds/logging.

#### Milestone 2: Single-Policy Self-Play Environment
1. Convert the mode-2 rules into `src/env/knight_self_play_env.py` exactly as described in the recipe:
   - Observation: `(3, board_size, board_size)` tensor with `my`, `opp`, `visited` channels.
   - Action space: `Discrete(8)` mapping to knight deltas.
   - Perspective swap: `reset`/`step` always return state from current player POV by swapping channels.
2. Implement action masking and illegal-move handling:
   - `info["legal_moves_mask"]` boolean array of shape `(8,)`.
   - Illegal choice ⇒ terminate with `reward=-1` and `info["illegal"]=True`.
3. Reward shaping hooks:
   - Terminal rewards ±1 for capture/no-moves.
   - Optional small step reward (e.g., `+0.001`) and mobility delta shaping `0.01 * (mobility_self - mobility_opp)` at non-terminal steps.
   - Parameterize shaping weights inside env config for quick sweeps.
4. Add vector-friendly factory in `src/training/env_factory.py` returning Gymnasium env wrapped for Stable-Baselines3 (e.g., `DummyVecEnv` with mask-aware wrapper).
5. Tests:
   - `tests/test_knight_env.py` verifying observation symmetry, action mask correctness, reward signs on terminal states, and deterministic resets.

#### Milestone 3: PPO Self-Play Training Loop
1. `src/players/rl/ppo_self_play.py`: thin wrapper that loads/saves Stable-Baselines3 PPO and exposes `select_action(observation, legal_mask)` for GUI/eval.
2. `src/training/ppo_self_play_train.py`:
   - Loads `config.yaml` (board size, shaping weights, PPO hyperparameters).
   - Instantiates `DummyVecEnv` via `env_factory`.
   - Configures PPO with `CnnPolicy`, `gamma=0.99`, `n_steps≈2048`, `batch_size=64`, `learning_rate=3e-4` (tweak via config).
   - Integrates legal-action masking by wrapping policy logits (either custom SB3 policy or callback).
   - Adds TensorBoard logging and periodic model checkpoints in `data/models/`.
3. Evaluation utilities:
   - `src/evaluation/self_play_eval.py`: run PPO vs random/greedy, compute win rates, push CSV metrics plus TensorBoard scalars.
   - CLI to load a saved PPO zip and pit against greedy to verify learning progress (baseline metric: PPO should exceed greedy within N timesteps).
4. Documentation updates:
   - `README.md` quickstart for training/eval.
   - `documentation/rules_en.md` cross-link to env assumptions (self-play perspective, illegal move penalties).

#### Milestone 4: Advanced Training, GUI Integration, and Multi-Mode Rollout
1. Optuna integration (optional but ready): `optuna_tune.py` wraps the PPO training loop with search spaces for shaping weights, learning rate, etc.
2. Enhance GUI to load PPO checkpoints:
   - Dropdown includes `ppo_self_play` wrapper; requires env-compatible observation builder for GUI matches.
   - Add toggle to show legal move mask and greedy recommendation for debugging.
3. Expand evaluation suite (`src/evaluation/eval_script.py`) to auto-discover available bots (random, greedy, PPO checkpoints) and log round-robin stats.
4. Dashboard (`src/gui/dashboard.py`) visualizes training curves, win matrices, illegal-move rates.
5. Once mode 2 pipeline (env + PPO self-play + greedy baseline) is stable, port the same env abstraction to modes 1 and 3 (shared env base class with mode-specific rules) and repeat training/eval steps.