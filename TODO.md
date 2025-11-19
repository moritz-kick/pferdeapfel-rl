Milestone 1:
Bugs:
- [x] When the game is over, the board is not updated to show the winner.
- [x] Winning condition is not working, when a player has no legal moves, the game should end.
- [x] its not possible to put apples on the tiles where the horse can jump to, since they are blocked for the jumping logic.
--- New:
- [ ] Redo and adjust python (backend/frontend) to support the new rules.
- [ ] Fix the UI to support the new rules and add a button to change the mode.
- [ ] In Classic, keep track of the winning score points/draw result inside the ui as well as in the logs.
- [ ] Add Keys and visual feedback in the gui. Keys:
    - [ ] h     - help
    - [ ] 1,2,3 - change mode
    - [ ] tab   - switch between horse moves
    - [ ] enter - place apple
    - [ ] esc   - undo
    - [ ] l     - log mode enabled/disabled