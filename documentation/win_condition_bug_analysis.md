# Win Condition Bug Analysis and Fix

## The Bug

In `src/game/game.py`, the `get_legal_moves()` method had a bug where it called `Rules.check_win_condition()` without passing the `last_mover` parameter when a player had no legal moves.

### Buggy Code (Before Fix)

```python
def get_legal_moves(self) -> list[tuple[int, int]]:
    """Get legal moves for the current player."""
    legal_moves = Rules.get_legal_knight_moves(self.board, self.current_player)
    
    if not legal_moves and not self.game_over:
        # BUG: Calls without last_mover parameter
        self.winner = Rules.check_win_condition(self.board)  # ❌ Missing last_mover!
        self.game_over = True
    
    return legal_moves
```

### Why This Is a Bug

In Mode 1 and Mode 2, when a capture occurs (both players on the same square), the winner is determined by who made the capture move (`last_mover`). The `check_win_condition` function needs this information:

```python
def check_win_condition(board: Board, last_mover: Optional[str] = None) -> Optional[str]:
    if board.mode in [1, 2]:
        # 1. Capture (Immediate Win)
        if board.white_pos == board.black_pos:
            return last_mover  # ❌ Returns None if last_mover is None!
        # ... rest of logic
```

When `last_mover` is `None`:
- If there's a capture, the function returns `None` instead of the correct winner
- The code then falls through to immobilization checks, which may give an incorrect result
- This can cause the wrong player to be declared the winner

### The Fix

When `get_legal_moves()` detects that the current player has no legal moves, the opponent must have been the last mover (since the current player can't move, the opponent must have just moved). The fix correctly passes the opponent as `last_mover`:

```python
def get_legal_moves(self) -> list[tuple[int, int]]:
    """Get legal moves for the current player."""
    legal_moves = Rules.get_legal_knight_moves(self.board, self.current_player)
    
    if not legal_moves and not self.game_over:
        # FIX: When current player has no moves, opponent was the last mover
        opponent = "black" if self.current_player == "white" else "white"
        self.winner = Rules.check_win_condition(self.board, last_mover=opponent)  # ✅ Correct!
        self.game_over = True
    
    return legal_moves
```

## Impact

This bug could cause incorrect winner determination in scenarios where:
1. A capture occurs and the game ends because the next player has no moves
2. The capture happens on the last move before a player gets stuck

The bug is particularly noticeable in self-play scenarios and when comparing HeuristicPlayer vs RandomPlayer, as shown in the terminal output where different branches produced different win rates.

## Testing

To verify the fix and find games where the bug manifests:

1. **Find divergent games**: Run `scripts/find_and_analyze_divergence.py`
   - This script plays games and compares buggy vs fixed logic
   - It will identify games where outcomes differ
   - Includes self-play scenarios as requested

2. **Test the fix**: Run `scripts/test_win_condition_fix.py`
   - Tests capture scenarios
   - Tests immobilization scenarios
   - Verifies correct winner determination

## Verification

The fix ensures that:
- ✅ Capture scenarios correctly identify the winner (the player who made the capture)
- ✅ Immobilization scenarios correctly identify the winner (the opponent of the stuck player)
- ✅ All calls to `check_win_condition` now properly pass `last_mover` when available

## Related Code Locations

- `src/game/game.py:78` - `make_move()` correctly passes `last_mover=self.current_player` ✅
- `src/game/game.py:144` - `get_legal_moves()` now correctly passes `last_mover=opponent` ✅ (fixed)
- `src/players/greedy.py:71` - Also correctly passes `last_mover` ✅
