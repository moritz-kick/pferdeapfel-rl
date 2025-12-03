# README.md

## Pferdeäpfel (Horse Apples) - Game Rules

Pferdeäpfel is a two-player abstract strategy game designed by Alex Randolph in 1981. It is played on an 8x8 grid, with players controlling horse figures that move like chess knights. The game involves placing obstacles ("horse apples") to restrict the opponent's movement. The original game has asymmetric goals: one player (White, the "escapee") tries to prolong the game to force the use of golden apples while the other (Black, the "chaser") aims to immobilize White before golden apples are placed on the board. We have added 2 additional more simplified modes to the game, which make it a symmetric game. For all modes we need the following components and setup:

### Components
- **Board**: An 8x8 grid (like a chessboard without colors).
- **Horses**: Two large plastic horse figures (one for each player), each with a hollow interior to hold apples.
- **Apples**: 40 horse apples in two colors – **28 Brown** (first phase) and **12 Golden** (second phase).

### Setup
- Players place their horses in opposite corners of the board (e.g., White at (0,0), Black at (7,7)).
- **Initial Drop**: Each player drops one brown apple on their starting square (marking it as visited).
- **White moves first.**

### Modes
- **Mode 1 – Free Placement (symmetric)**: At the start of every turn, the active player **must** drop a brown apple on any empty square. After that compulsory placement, the player makes a legal knight move. There are no golden apples or scoring phases and both players can capture. The first horse that is captured or has no legal moves on its own turn loses.
- **Mode 2 – Trail Placement (symmetric)**: The player simply moves their knight to a legal square. Immediately after the move, the square that was just vacated is filled with a brown apple (leaving a “trail”). No extra placements exist, the apples are colorless blockers, and captures remain legal for both sides. A side that becomes immobile on its turn loses.
- **Mode 3 – Classic (original asymmetric rules)**: Before moving, the active player places an apple on the square they are leaving (brown until the supply is empty, then golden). After moving they may optionally drop a second apple on any empty square, as long as it does **not** remove White’s final escape. Only Black is allowed to capture; White focuses on survival. Black tries to capture or trap White before the 28 brown apples run out, while White tries to force the golden phase and collect points.

### Win Conditions
- **Mode 1**: Immediate win for the player who captures the opponent or leaves them without legal knight moves.
- **Mode 2**: Identical to Mode 1 — capture or immobilize the opponent to win.
- **Mode 3**: Scoring follows the original rules.
  - **Black win condition**: Capture White or immobilize White *before* the brown supply is exhausted. Each unused brown apple is worth 1 point for Black.
  - **White win condition**: The instant a golden apple is required (because the brown supply is empty), White wins the match but play continues for scoring. Every golden apple placed scores 1 point for White. White also earns **24 points** if either (a) all 12 golden apples are used and White can still move or (b) Black becomes immobilized during the golden phase.
  - **Draw**: If Black captures White on the same turn the very last brown apple was used for the mandatory placement (before any golden apple hits the board), the game is a draw.


For development details, see [project_plan.md](documentation/project_plan.md).
For detailed rules in german, refer to [original_rules_de.md](documentation/original_rules_de.md).
For detailed rules in english with the differened modes, refer to [rules_en.md](documentation/rules_en.md).

## Training

### Basic PPO Training (against random opponent)

```bash
# Train PPO agent against random opponents
uv run python -m src.training.train_ppo --steps 10_000_000
```

### AlphaZero-Style Self-Play Training

The self-play training implements an AlphaZero-like training loop where the agent continuously improves by playing against its best version:

1. Train the current model by playing against the best model
2. Periodically evaluate current vs best (1000 games)
3. If current wins ≥ 55%, current becomes the new best
4. Repeat

```bash
# Start fresh self-play training
uv run python scripts/train_self_play.py

# Start from a model pretrained against random (recommended)
uv run python scripts/train_self_play.py --from-pretrained data/models/ppo/best_model/best_model.zip

# Continue interrupted training
uv run python scripts/train_self_play.py --continue

# Custom settings
uv run python scripts/train_self_play.py \
    --n-envs 32 \
    --steps 100000000 \
    --eval-freq 200000 \
    --n-eval-games 200 \
    --win-threshold 0.55
```

### Benchmarking Models

Compare trained models against various opponents:

```bash
# Full benchmark (PPO vs Random, Greedy, other PPO models)
uv run python scripts/benchmark_self_play.py

# Quick benchmark (10 games per matchup)
uv run python scripts/benchmark_self_play.py --quick
```
