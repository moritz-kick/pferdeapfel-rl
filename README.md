# README.md

## Pferdeäpfel (Horse Apples) - Game Rules

Pferdeäpfel is a two-player abstract strategy game designed by Alex Randolph in 1981. It is played on an 8x8 grid, with players controlling horse figures that move like chess knights. The game involves placing obstacles ("horse apples") to restrict the opponent's movement. The game has asymmetric goals: one player (White, the "escapee") tries to prolong the game to force the use of golden apples, while the other (Black, the "chaser") aims to immobilize White before golden apples are placed on the board.

### Components
- **Board**: An 8x8 grid (like a chessboard without colors).
- **Horses**: Two large plastic horse figures (one for each player), each with a hollow interior to hold apples.
- **Apples**: 40 horse apples in two colors – brown (first phase) and golden (second phase). Assume 20 brown and 20 golden for implementation.

### Setup
- Players place their horses in opposite corners of the board (e.g., White at (0,0), Black at (7,7)).
- Each horse starts with one brown apple inside it.
- The remaining apples are set aside in a shared pool (brown first, then golden when brown are depleted).
- A random player starts (determined at game start).

### Gameplay
Players alternate turns. On each turn:
1. **Move the Horse**: Move your horse like a chess knight (L-shape: 2 in one direction, 1 perpendicular). The landing square must be empty (no apple or opponent's horse). Knights can jump over occupied squares.
   - Upon moving, any apple inside the horse drops onto the vacated square, blocking it permanently.
2. **Add an Apple to the Horse**: After moving, place one apple from the pool (brown if available, else golden) into your horse.
3. **Optional Extra Placement**: Before ending your turn, you may place one additional apple from the pool directly onto any empty square on the board.
   - This placement **cannot** immediately leave the White player with no legal moves. If it would, the placement is invalid and must be skipped or adjusted.
   - Placing an extra apple uses another apple from the pool.

- When brown apples are depleted, switch to golden apples. The first use of a golden apple (either added to a horse or placed extra) marks the "golden phase."
- Blocked squares (with apples) cannot be landed on but do not block jumps.

### Game End
- The game ends when White cannot make a legal move on their turn.
- **White Wins**: If at least one golden apple has been placed on the board (or dropped via move).
- **Black Wins**: If no golden apples are on the board.

### Notes
- The game emphasizes strategy: White aims to survive long enough to deplete brown apples, while Black blocks White's knight moves aggressively.
- Apples inside horses are always dropped on move (typically 1 per turn, as one is added each turn).
- Total apples: 40, but games may end earlier.
- This implementation will focus on digital simulation, ignoring physical aspects like apples rolling.

For development details, see [project_plan.md](project_plan.md).