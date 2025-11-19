# Pferdeäpfel – Modes of Play

This project now supports three distinct modi (game modes). Each mode defines how horses may place apples and what counts as a win. In all modes:

- The board is an `8×8` grid with White starting in the top-left corner `(0, 0)` and Black in the bottom-right `(7, 7)`.
- Horses always move like chess knights. If a destination is occupied, that move is illegal.
- White moves first. To keep experiments fair, randomly assign algorithms to White/Black **before** a game starts (e.g., during training or evaluation) so that both get equal exposure to playing first.

## Mode 1 – Free Placement

1. At the start of every turn, the active player must place a plain horse apple on **any** empty square.
2. After placing the apple, the player makes a legal knight move.
3. There are no color phases and no golden apples. Apples are identical blockers.
4. The first horse that cannot move on its turn loses the game immediately, regardless of how many apples are on the board.

This mode emphasizes planning traps through proactive blocking.

## Mode 2 – Trail Placement

1. On each turn, the player simply moves their horse to a legal square.
2. After jumping, the square the horse just left is automatically filled with a horse apple (“leave a trace”).
3. No extra placements exist and apples have no colors.
4. As in Mode 1, the horse that first has zero legal knight moves on its turn loses.

This mode is about survival while leaving an unavoidable trail behind.

## Mode 3 – Classic

This mode recreates the original game (*Pferdeäppel* by Bütehorn Spiele) with its asymmetric win conditions and scoring.

### Components & Setup
- **Apples:** 28 Brown, 12 Golden.
- **Setup:** 
  - Each player drops a brown apple on their starting square.
  - White moves first.

### Turn Sequence
1. **Move:** Make a legal knight move.
2. **Mandatory Placement:** After the move, place a brown apple on the square where the turn ended (the horse's current position).
   - *Note:* In the physical game, you put an apple into the horse, and it drops onto the square. This means every visited square (including start) is marked with an apple.
3. **Optional Placement:** You may place **one additional brown apple** on any empty square.
   - **Restriction:** You cannot place this extra apple if it blocks White's last escape route.

### Win Conditions & Phases

**Phase 1: Brown Apples (The Chase)**
- Brown (Black) tries to catch White or immobilize White.
- **Brown Wins Immediately If:**
  - Brown captures White (lands on White's square).
  - White is immobilized (cannot move).
  - *Condition:* This must happen **before** the supply of 28 brown apples is exhausted.
- **Scoring for Brown:** If Brown wins, they get **1 point** for every unused brown apple remaining in the supply.

**Phase 2: Golden Apples (The Escape)**
- **White Wins the Game** as soon as the first **Golden Apple** is brought into play (i.e., when the brown supply is empty and a player must draw a golden apple for mandatory or optional placement).
- **However, the game continues for scoring!**
- Brown continues trying to catch White.
- Every Golden Apple placed on the board counts as **1 point for White**.

**End of Golden Phase:**
- The game ends when:
  - Brown catches White (Game over, White keeps points accumulated so far).
  - White is immobilized (Game over, White keeps points).
  - **Brown is immobilized:** White gets **24 points**.
  - **All 12 Golden Apples are used** and White can still move: White gets **24 points**.

**Draw Condition:**
- The game ends in a **Draw** if Brown catches White **exactly when** all 28 brown apples have been used, but **before** any golden apple has been brought into play.



