#!/usr/bin/env python
"""Quick sanity check for Mode 1 after the rule change."""

from src.env.knight_self_play_env import KnightSelfPlayEnv
import numpy as np

def test_mode1():
    """Test Mode 1 works correctly with the new move-then-place rules."""
    env = KnightSelfPlayEnv(mode=1)
    
    successes = 0
    failures = 0
    
    for episode in range(100):
        obs, info = env.reset()
        done = False
        steps = 0
        episode_ok = True
        
        while not done and steps < 100:
            mask = env.action_masks()
            move_mask = mask[:8]
            apple_mask = mask[8:]
            
            valid_moves = np.where(move_mask)[0]
            valid_apples = np.where(apple_mask)[0]
            
            if len(valid_moves) == 0 or len(valid_apples) == 0:
                print(f'Episode {episode}: No valid actions at step {steps}!')
                episode_ok = False
                break
            
            move = np.random.choice(valid_moves)
            apple = np.random.choice(valid_apples)
            
            obs, reward, done, truncated, info = env.step(np.array([move, apple]))
            
            # Check for invalid action errors
            if 'error' in info:
                print(f'Episode {episode}: Error at step {steps}: {info["error"]}')
                episode_ok = False
                break
            
            steps += 1
        
        if episode_ok:
            successes += 1
            winner = info.get('winner', 'N/A')
            print(f'Episode {episode}: OK - {steps} steps, winner={winner}, agent={env.agent_color}')
        else:
            failures += 1
    
    print(f'\n=== Results: {successes}/100 episodes succeeded, {failures} failed ===')
    return failures == 0

if __name__ == '__main__':
    success = test_mode1()
    exit(0 if success else 1)
