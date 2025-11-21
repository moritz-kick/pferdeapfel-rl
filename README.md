# README.md

## Pferdeäpfel (Horse Apples) - Game Rules

Pferdeäpfel is a two-player abstract strategy game designed by Alex Randolph in 1981. It is played on an 8x8 grid, with players controlling horse figures that move like chess knights. The game involves placing obstacles ("horse apples") to restrict the opponent's movement. The game has asymmetric goals: one player (White, the "escapee") tries to prolong the game to force the use of golden apples, while the other (Black, the "chaser") aims to immobilize White before golden apples are placed on the board.

### Components
- **Board**: An 8x8 grid (like a chessboard without colors).
- **Horses**: Two large plastic horse figures (one for each player), each with a hollow interior to hold apples.
- **Apples**: 40 horse apples in two colors – **28 Brown** (first phase) and **12 Golden** (second phase).

### Setup
- Players place their horses in opposite corners of the board (e.g., White at (0,0), Black at (7,7)).
- **Initial Drop**: Each player drops one brown apple on their starting square (marking it as visited).
- The remaining apples are set aside in a shared pool.
- **White moves first.**

### Gameplay
Players alternate turns. On each turn:
1. **Move the Horse**: Move your horse like a chess knight. The landing square must be empty.
2. **Mandatory Placement**: After moving, place a brown apple on the square you just landed on.
   - *Effect*: Every square a horse visits (ends a turn on) becomes blocked for the rest of the game.
3. **Optional Extra Placement**: You may place **one additional brown apple** from the pool onto any empty square.
   - **Restriction**: You cannot use this to block White's last escape route.

### Win Conditions & Phases

**Phase 1: Brown Apples (The Chase)**
- **Brown (Black) Wins Immediately** if they catch White or White cannot move, **before** the brown apples run out.
- **Scoring**: Brown gets 1 point for every unused brown apple.

**Phase 2: Golden Apples (The Escape)**
- **White Wins the Game** as soon as the first **Golden Apple** is brought into play (when brown apples run out).
- **Scoring Continues**: The game continues until Brown catches White or someone is immobilized.
- White gets **1 point** for every Golden Apple placed.
- If White survives all 12 Golden Apples or Brown gets stuck, White gets **24 points**.

**Draw:**
- The game is a **Draw** if Brown catches White **after** all brown apples are used but **before** a golden apple enters play.


### Notes
- The game emphasizes strategy: White aims to survive long enough to deplete brown apples, while Black blocks White's knight moves aggressively.

## RL (Milestone 2)
- Gymnasium environment for mode 2 lives at `src/env/pferdeapfel_env.py` (Discrete(64) actions, flattened observations with positions/turn/flags).
- PPO training config is in `src/training/config.yaml`; run training with `python -m src.training.ppo_train` (models saved to `data/models/`).
- To add opponent variety during self-play training, set `env.random_opponent_chance` (e.g., `0.30`) to play a fraction of episodes against a random legal-move bot instead of the frozen snapshot.
- Self-play tournaments pit the current model against the last 4 snapshots; the best performer becomes the “best” opponent. During training episodes, opponents are sampled with 40% best snapshot, 30% random old snapshot (pool size 4), otherwise the live policy (plus any random games if `random_opponent_chance` > 0).
- Unless `trainer.initial_model_path` is provided, training auto-loads the most recently saved PPO model under `data/models/ppo_pferdeapfel` as starting weights.
- RL players in the GUI: `dqn` plus a PPO option that auto-loads the latest model from `data/models/`.
- Evaluate an RL model against the random baseline with `python -m src.evaluation.eval_script --model <path> --games 20`; results are written to `src/evaluation/results.csv`.
- `src/gui/dashboard.py` plots win rates from the evaluation CSV.
- Training writes TensorBoard logs to `data/logs/rl`; launch the dashboard with `tensorboard --logdir data/logs/rl`.
- Generate a short, LLM-friendly summary of a run with `python -m src.training.tb_summary --logdir data/logs/rl --format text` (use `--output summary.txt` to save).
- RL I/O smoke tests and logs land in `tests/test_rl_io.py` and `data/logs/rl_player/`.

For development details, see [project_plan.md](documentation/project_plan.md). For detailed rules in german, refer to [original_rules_de.md](documentation/original_rules_de.md). For detailed rules in english with the differened modes, refer to [rules_en.md](documentation/rules_en.md).
