# Pferdeäpfel – Modes of Play

This project now supports three distinct modi (game modes). Each mode defines how horses may place apples and what counts as a win. In all modes:

- The board is an `8×8` grid with White starting in the top-left corner `(0, 0)` and Black in the bottom-right `(7, 7)`.
- Horses always move like chess knights. If a destination is occupied, that move is illegal.
- White moves first. To keep experiments fair, randomly assign algorithms to White/Black **before** a game starts (e.g., during training or evaluation) so that both get equal exposure to playing first.

## Mode 1 – Free Placement (Move First)

1. The active player makes a legal knight move.
2. After moving, the player must place a plain horse apple on **any** empty square (including the square they just left).
3. There are no color phases and no golden apples. Apples are identical blockers.
4. The first horse that cannot move on its turn loses the game immediately, regardless of how many apples are on the board.
5. Captures are allowed for both players and win the game.

This mode emphasizes planning traps through proactive blocking. The move-first order ensures the action space dimensions are independent.

## Mode 2 – Trail Placement

1. On each turn, the player simply moves their horse to a legal square.
2. After jumping, the square the horse just left is automatically filled with a horse apple (“leave a trace”).
3. No extra placements exist and apples have no colors.
4. As in Mode 1, the horse that first has zero legal knight moves on its turn loses.
5. Captures are allowed for both players and win the game.

This mode is about survival while leaving an unavoidable trail behind.

## Mode 3 – Classic

This mode recreates the original game (*Pferdeäppel*) with its asymmetric win conditions and scoring.

### Components & Setup
- **Apples:** 28 Brown, 12 Golden.
- **Setup:**
  - Each player puts the knights on their starting squares.
  - White moves first.

### Turn Sequence
To ensure logical consistency regarding the "Draw" condition, the turn structure is strictly defined as follows (and only Black is allowed to capture; White landing on Black’s square is illegal):

1. **Mandatory Placement:**
   Before moving, the active player places one brown apple on the square their horse currently occupies.
2. **Move:**
   The player makes a legal knight move to a new square.
3. **Optional Placement:**
   The player may choose to place **one additional brown apple** on any empty square on the board.
   - **Restriction:** You cannot place this extra apple if it blocks White's last remaining escape route.

*(Note: If the supply of brown apples runs out, players must use golden apples for placements. More about that in Phase 2.)*

### Win Conditions & Phases

**Phase 1: Brown Apples (The Chase)**
- Black tries to catch White or immobilize White.
- **Black Wins Immediately If:**
  - Black captures White (lands on White's square).
  - White is immobilized (cannot move).
  - *Condition:* This must happen **before** the supply of 28 brown apples is exhausted.
- **Scoring for Black:** If Black wins, they get **1 point** for every unused brown apple remaining in the supply.
- **White Bonus:** If Black becomes immobilized **before** any golden apples are placed (i.e., still in the brown phase),
  White wins immediately and is awarded **12 points**—the value of the untouched golden reserve—even though Phase 2 never begins.

**The Draw Condition:**
- The game ends in a **Draw** if Black captures White on the exact turn where the **last brown apple** was used for the **Mandatory Placement**.
- *Logic:* The supply hits 0 during step 1 (Mandatory), and the catch happens in step 2 (Move). No Golden Apple was needed, but no Brown Apples remain for points.

**Phase 2: Golden Apples (The Escape)**
- **White Wins the Game** as soon as a **Golden Apple** is required for play (i.e., the brown supply is empty, and a player must perform a Mandatory or Optional placement using a golden apple).
- **Scoring Mode:** Although White has technically won the match, play continues to determine the score. Every Golden Apple placed on the board counts as **1 point for White**.

**End of Golden Phase:**
- The game ends when:
  - Black catches White (Game over, White keeps points accumulated so far).
  - White is immobilized (Game over, White keeps points).
  - Black is immobilized: (Game over, White keeps points).
  - **All 12 Golden Apples are used** and White can still move: White gets **24 points**.