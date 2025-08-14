# DuoPolo_simulation

Pololu Search Simulator — Quick Guide
=====================================

What this is
------------
A fast, configurable simulator for your Pololu grid search system that mirrors your MQTT topics and your A* + reward logic with clue-biased probability updates.

Key features
------------
- In-process MQTT bus (no broker needed), using topics like `00/status`, `00/clue`, `00/alert`.
- 2 robots by default:
  - '00' starts at (0,0), facing North.
  - '01' starts at (9,9), facing South.
- Grid world (default 10x10). Clues are placed with a 1/(1+Manhattan distance) bias around the kid.
- Batch mode for high-speed experiments and statistics.
- Live viewer (pygame) with pause/fast/reset.

Install (locally)
-----------------
- Python 3.9+ recommended
- `pip install pygame matplotlib`

Run batch (e.g., 1000 episodes)
-------------------------------
    python pololu_search_sim.py --mode batch --episodes 1000 --grid 10 --seed 42 --out sim_out

Outputs:
- `sim_out/summary.json` — found rate, avg steps, avg revisits/path length.
- `sim_out/episodes.csv` — per-episode metrics and ground truth.

Run live viewer
---------------
    python pololu_search_sim.py --mode view --grid 10 --seed 123

Controls:
- SPACE: pause/resume
- F:     fast mode toggle
- R:     reset episode

Tuning knobs
------------
Edit `Config` in the script (top) to change:
- `grid_size`, `step_seconds`, `CLUE_COUNT`
- `CENTER_STEP`, `SWITCH_COL_BASE`, `REWARD_GAIN`
- `reward_factor_astar` (keeps A* edges positive)

Keeping it in sync with your robot code
---------------------------------------
The planner/reward logic is encapsulated inside `RobotAgent`:
- `_update_prob_reward()` matches your clue bump and reward scaling.
- `_pick_goal()` mirrors your pre-clue column bias.
- `_a_star()` uses turn cost, serpentine cost, and tempered reward.

As your on-robot code evolves, keep these functions aligned. If you want, we can factor these into a small shared module that both the sim and robot import to completely eliminate drift.
