
#!/usr/bin/env python3
# Pololu Multi-Robot Grid Search Simulator
# - In-process MQTT-like bus emulation using your topics
# - 2D line grid world (default 10x10), easily configurable
# - Clues biased near the kid by 1/(1+Manhattan distance)
# - A* policy mirroring your reward update and pre-clue serpentine bias
# - Two modes:
#     * batch: run N episodes and write summary CSVs (fast)
#     * view : live viewer using pygame with step/fast-forward controls
#
# Start poses (per your request):
#   Robot '00' at (0,0) facing North; Robot '01' at (size-1,size-1) facing South.
#
# Usage (examples):
#   python pololu_search_sim.py --mode batch --episodes 1000 --grid 10 --seed 42
#   python pololu_search_sim.py --mode view  --grid 10 --seed 123
#
# Requirements: pygame (for view mode), matplotlib (optional for saving plots)
#   pip install pygame matplotlib
#
from __future__ import annotations
import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional

# ------------------------
# CONFIG dataclass
# ------------------------
@dataclass
class Config:
    grid_size: int = 10
    step_seconds: float = 2.0
    seed: Optional[int] = 42
    robots: List[str] = field(default_factory=lambda: ["00", "01"])
    CENTER_STEP: float = 0.7
    SWITCH_COL_BASE: float = 0.2
    INTENT_TTL_STEPS: int = 1
    INTENT_PENALTY: float = 8.0
    REVISIT_PENALTY: float = 4.0
    REWARD_GAIN: float = 5.0
    CLUE_COUNT: int = 3
    max_steps_factor: int = 5  # grid^2 * factor
    reward_factor_astar: float = 0.2  # temper reward to keep edges non-negative
    viewer_fps: int = 8

# ------------------------
# MQTT-like bus (in-process)
# ------------------------
class InProcBus:
    def __init__(self):
        # subscribers keyed by topic prefix
        self.subs: Dict[str, List[Callable[[str, str], None]]] = {}

    def subscribe(self, topic_prefix: str, fn: Callable[[str, str], None]):
        self.subs.setdefault(topic_prefix, []).append(fn)

    def publish(self, topic: str, payload: str):
        for prefix, handlers in self.subs.items():
            if topic.startswith(prefix):
                for h in handlers:
                    h(topic, payload)

# ------------------------
# World
# ------------------------
EMPTY = 0
VISITED = 2
DIRS = [(0,1),(1,0),(0,-1),(-1,0)]  # N,E,S,W

@dataclass
class World:
    size: int
    kid: Tuple[int, int]
    clues: List[Tuple[int, int]]
    grid: List[List[int]] = field(init=False)
    def __post_init__(self):
        self.grid = [[EMPTY for _ in range(self.size)] for _ in range(self.size)]

def sample_kid_and_clues(size: int, n_clues: int, rng: random.Random):
    kid = (rng.randrange(size), rng.randrange(size))
    cells = [(x,y) for y in range(size) for x in range(size) if (x,y) != kid]
    def w(cell):
        return 1.0 / (1 + abs(cell[0]-kid[0]) + abs(cell[1]-kid[1]))
    weights = [w(c) for c in cells]
    s = sum(weights)
    weights = [wi/s for wi in weights]
    clues = []
    for _ in range(n_clues):
        c = rng.choices(cells, weights=weights, k=1)[0]
        clues.append(c)
        i = cells.index(c)
        cells.pop(i); weights.pop(i)
        if weights:
            s = sum(weights); weights[:] = [wi/s for wi in weights]
    return kid, clues

# ------------------------
# Robot agent
# ------------------------
@dataclass
class RobotAgent:
    robot_id: str
    other_id: str
    world: World
    bus: InProcBus
    prefers_left: bool
    start_pos: Tuple[int,int]
    start_heading: Tuple[int,int]
    cfg: Config

    # state
    pos: Tuple[int,int] = field(init=False)
    heading: Tuple[int,int] = field(init=False)
    grid: List[List[int]] = field(init=False)
    prob_map: List[List[float]] = field(init=False)
    reward_map: List[List[float]] = field(init=False)
    clues: List[Tuple[int,int]] = field(default_factory=list)
    first_clue_seen: bool = field(default=False)
    other_intent: Optional[Tuple[int,int]] = field(default=None)
    other_intent_ttl: int = field(default=0)
    path_history: List[Tuple[int,int]] = field(default_factory=list)
    revisits: int = field(default=0)
    steps: int = field(default=0)
    found_object: bool = field(default=False)

    def __post_init__(self):
        self.pos = self.start_pos
        self.heading = self.start_heading
        self.grid = [[EMPTY for _ in range(self.world.size)] for _ in range(self.world.size)]
        self.prob_map = [[0.0 for _ in range(self.world.size)] for _ in range(self.world.size)]
        self.reward_map = [[0.0 for _ in range(self.world.size)] for _ in range(self.world.size)]
        self.grid[self.pos[1]][self.pos[0]] = VISITED
        self.world.grid[self.pos[1]][self.pos[0]] = VISITED
        self.path_history.append(self.pos)
        # subscribe to messages from the other robot
        self.bus.subscribe(f"{self.other_id}/", self._on_other_msg)
        self._pub_status_pos()

    def _pub_status_pos(self):
        x,y = self.pos; hx,hy = self.heading
        self.bus.publish(f"{self.robot_id}/status", f"pos:{x},{y};heading:{hx},{hy}")
    def _pub_intent(self, cell):
        x,y = cell; self.bus.publish(f"{self.robot_id}/status", f"intent:{x},{y}")
    def _pub_clue(self, cell):
        x,y = cell; self.bus.publish(f"{self.robot_id}/clue", f"clue:{x},{y}")
    def _pub_alert_object(self, cell):
        x,y = cell; self.bus.publish(f"{self.robot_id}/alert", f"object:{x},{y}")

    def _on_other_msg(self, topic: str, payload: str):
        """Handle bus messages from the other robot."""
        try:
            sender, sub_topic = topic.split("/", 1)
        except ValueError:
            return
        if sender != self.other_id:
            return
        if sub_topic == "status" and payload.startswith("intent:"):
            xy = payload.split("intent:", 1)[1]
            try:
                x, y = [int(v) for v in xy.split(",")]
            except ValueError:
                return
            self.other_intent = (x, y)
            self.other_intent_ttl = self.cfg.INTENT_TTL_STEPS
        elif sub_topic == "alert" and payload.startswith("object:"):
            self.found_object = True
        elif sub_topic == "clue" and payload.startswith("clue:"):
            xy = payload.split("clue:", 1)[1]
            try:
                x, y = [int(v) for v in xy.split(",")]
            except ValueError:
                return
            if (x, y) not in self.clues:
                self.clues.append((x, y))
                self.first_clue_seen = True

    def _edge_distance_from_side(self, x: int) -> int:
        size = self.world.size
        return x if self.prefers_left else (size-1-x)

    def _centerward_step_cost(self, curr_x: int, next_x: int) -> float:
        if self.first_clue_seen: return 0.0
        if next_x == curr_x:     return 0.0
        d_curr = self._edge_distance_from_side(curr_x)
        d_next = self._edge_distance_from_side(next_x)
        toward_center = (d_next < d_curr)
        cost = self.cfg.SWITCH_COL_BASE
        if toward_center:
            cost += self.cfg.CENTER_STEP * (d_curr - d_next)
        return cost

    def _update_prob_reward(self):
        size = self.world.size
        for y in range(size):
            for x in range(size):
                if self.grid[y][x] == VISITED:
                    self.prob_map[y][x] = 0.0
                    self.reward_map[y][x] = 0.0
                    continue
                base = 1/(size*size)
                clue_sum = 0.0
                for (cx,cy) in self.clues:
                    clue_sum += 5.0/(1 + abs(x-cx) + abs(y-cy))
                self.prob_map[y][x] = base + clue_sum
                self.reward_map[y][x] = self.prob_map[y][x] * self.cfg.REWARD_GAIN

    def _intent_active(self) -> bool:
        return (self.other_intent is not None) and (self.other_intent_ttl > 0)

    def _pick_goal(self) -> Tuple[int,int]:
        size = self.world.size
        best, best_val = None, -1e9
        for y in range(size):
            for x in range(size):
                if self.grid[y][x] != EMPTY: continue
                val = self.reward_map[y][x]
                if not self.first_clue_seen:
                    val -= 0.3 * self._edge_distance_from_side(x)
                if val > best_val:
                    best_val = val; best = (x,y)
        if best is None:
            unknown = [(x,y) for y in range(size) for x in range(size) if self.grid[y][x]==EMPTY]
            if not unknown: return self.pos
            best = min(unknown, key=lambda c: abs(c[0]-self.pos[0])+abs(c[1]-self.pos[1]))
        return best

    def _a_star(self, start, goal):
        import heapq
        size = self.world.size
        def neighbors(c):
            x,y = c
            for dx,dy in DIRS:
                nx,ny = x+dx, y+dy
                if 0 <= nx < size and 0 <= ny < size:
                    yield (nx,ny)
        def heuristic(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
        start_heading = self.heading
        frontier = [(0.0, start, start_heading)]
        came_from = {start: None}
        cost_so_far = {start: 0.0}
        RF = self.cfg.reward_factor_astar
        while frontier:
            _, current, cur_h = heapq.heappop(frontier)
            if current == goal: break
            for nx,ny in neighbors(current):
                move_cost = 1.0
                dx,dy = nx-current[0], ny-current[1]
                turn_cost = 1.0 if (dx,dy) != cur_h else 0.0
                serp_cost = self._centerward_step_cost(current[0], nx)
                intent_cost = self.cfg.INTENT_PENALTY if (self._intent_active() and (nx,ny)==self.other_intent) else 0.0
                revisit_cost = self.cfg.REVISIT_PENALTY if self.world.grid[ny][nx] == VISITED else 0.0
                reward_bonus = RF * self.reward_map[ny][nx]
                step_cost = max(0.01, (move_cost + turn_cost + serp_cost + intent_cost + revisit_cost) - reward_bonus)
                new_cost = cost_so_far[current] + step_cost
                if (nx,ny) not in cost_so_far or new_cost < cost_so_far[(nx,ny)]:
                    cost_so_far[(nx,ny)] = new_cost
                    priority = new_cost + heuristic((nx,ny), goal)
                    heapq.heappush(frontier, (priority, (nx,ny), (dx,dy)))
                    came_from[(nx,ny)] = current
        if goal not in came_from: return [start]
        path = [goal]; cur = goal
        while cur != start:
            cur = came_from[cur]; 
            if cur is None: break
            path.append(cur)
        path.reverse(); return path

    def _turn_towards(self, nxt):
        dx,dy = nxt[0]-self.pos[0], nxt[1]-self.pos[1]
        self.heading = (dx,dy)

    def step(self, global_stop: bool):
        if self.found_object or global_stop: return
        if self.other_intent_ttl > 0: self.other_intent_ttl -= 1
        self._update_prob_reward()
        goal = self._pick_goal()
        path = self._a_star(self.pos, goal)
        if len(path) < 2: return
        next_cell = path[1]
        self._pub_intent(next_cell)
        # Execute (no stochastic yield; deterministic via A* intent cost)
        self._turn_towards(next_cell)
        self.pos = next_cell
        self.steps += 1
        if self.world.grid[self.pos[1]][self.pos[0]] == VISITED or self.grid[self.pos[1]][self.pos[0]] == VISITED:
            self.revisits += 1
        self.grid[self.pos[1]][self.pos[0]] = VISITED
        self.world.grid[self.pos[1]][self.pos[0]] = VISITED
        self.path_history.append(self.pos)
        self._pub_status_pos()
        # Clues and kid
        if self.pos in self.world.clues and self.pos not in self.clues:
            self.clues.append(self.pos); self.first_clue_seen = True; self._pub_clue(self.pos)
        if self.pos == self.world.kid:
            self._pub_alert_object(self.pos); self.found_object = True

# ------------------------
# Single episode runner
# ------------------------
@dataclass
class EpisodeResult:
    found: bool
    steps: int
    sim_seconds: float
    revisits_by_robot: Dict[str,int]
    path_length_by_robot: Dict[str,int]
    kid: Tuple[int,int]
    clues: List[Tuple[int,int]]
    collisions: int

def run_episode(cfg: Config, seed: Optional[int]=None) -> EpisodeResult:
    rng = random.Random(cfg.seed if seed is None else seed)
    kid, clues = sample_kid_and_clues(cfg.grid_size, cfg.CLUE_COUNT, rng)
    world = World(size=cfg.grid_size, kid=kid, clues=clues)
    bus = InProcBus()
    agents: Dict[str, RobotAgent] = {}
    agents["00"] = RobotAgent(robot_id="00", other_id="01", world=world, bus=bus,
                              prefers_left=True, start_pos=(0,0), start_heading=(0,1), cfg=cfg)
    agents["01"] = RobotAgent(robot_id="01", other_id="00", world=world, bus=bus,
                              prefers_left=False, start_pos=(cfg.grid_size-1, cfg.grid_size-1), start_heading=(0,-1), cfg=cfg)
    # log subscriber (no-op speed)
    found = False; collisions = 0; current_step = 0
    max_steps = cfg.grid_size * cfg.grid_size * cfg.max_steps_factor
    global_stop = False
    while current_step < max_steps:
        current_step += 1
        order = ["00","01"]; rng.shuffle(order)
        prev = {rid: agents[rid].pos for rid in order}
        for rid in order:
            agents[rid].step(global_stop)
        # collision (both to same cell) — rare with intent cost but check
        if agents["00"].pos == agents["01"].pos and not (agents["00"].found_object or agents["01"].found_object):
            collisions += 1
            # revert lower id
            mover = agents["00"]
            if len(mover.path_history) >= 2:
                mover.path_history.pop()
                mover.pos = mover.path_history[-1]
                mover.revisits += 1
        if agents["00"].found_object or agents["01"].found_object:
            global_stop = True; found = True; break
    return EpisodeResult(
        found=found,
        steps=current_step,
        sim_seconds=current_step * cfg.step_seconds,
        revisits_by_robot={rid: agents[rid].revisits for rid in agents},
        path_length_by_robot={rid: len(agents[rid].path_history)-1 for rid in agents},
        kid=world.kid, clues=world.clues, collisions=collisions
    )

# ------------------------
# Batch runner
# ------------------------
def run_batch(cfg: Config, episodes: int, base_seed: Optional[int]) -> Dict:
    rng = random.Random(base_seed if base_seed is not None else cfg.seed)
    results = []
    for ep in range(episodes):
        seed = rng.randrange(10**9)
        res = run_episode(cfg, seed=seed)
        results.append(res)
    # aggregate
    found_count = sum(1 for r in results if r.found)
    avg_steps = sum(r.steps for r in results)/len(results)
    avg_revisits_00 = sum(r.revisits_by_robot["00"] for r in results)/len(results)
    avg_revisits_01 = sum(r.revisits_by_robot["01"] for r in results)/len(results)
    avg_path_00 = sum(r.path_length_by_robot["00"] for r in results)/len(results)
    avg_path_01 = sum(r.path_length_by_robot["01"] for r in results)/len(results)
    summary = {
        "episodes": episodes,
        "found_rate": found_count/episodes,
        "avg_steps": avg_steps,
        "avg_revisits_00": avg_revisits_00,
        "avg_revisits_01": avg_revisits_01,
        "avg_path_00": avg_path_00,
        "avg_path_01": avg_path_01,
    }
    return {"summary": summary, "results": results}

# ------------------------
# Live Viewer (pygame)
# ------------------------
def run_viewer(cfg: Config, seed: Optional[int] = None):
    try:
        import pygame
    except Exception as e:
        print("Viewer requires pygame. Install via: pip install pygame")
        raise
    pygame.init()
    cell = 48
    size = cfg.grid_size
    W,H = size*cell, size*cell
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    # set up one episode
    res = None
    rng = random.Random(cfg.seed if seed is None else seed)
    kid, clues = sample_kid_and_clues(cfg.grid_size, cfg.CLUE_COUNT, rng)
    world = World(size=cfg.grid_size, kid=kid, clues=clues)
    bus = InProcBus()
    a0 = RobotAgent("00","01", world, bus, True, (0,0), (0,1), cfg)
    a1 = RobotAgent("01","00", world, bus, False,(size-1,size-1),(0,-1), cfg)
    agents = {"00": a0, "01": a1}
    font = pygame.font.SysFont("Arial", 16)
    running = True; paused=False; fast=False; step=0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: paused = not paused
                if event.key == pygame.K_f: fast = not fast
                if event.key == pygame.K_r:  # reset
                    kid, clues = sample_kid_and_clues(cfg.grid_size, cfg.CLUE_COUNT, rng)
                    world = World(size=cfg.grid_size, kid=kid, clues=clues)
                    a0 = RobotAgent("00","01", world, bus, True, (0,0), (0,1), cfg)
                    a1 = RobotAgent("01","00", world, bus, False,(size-1,size-1),(0,-1), cfg)
                    agents = {"00": a0, "01": a1}
                    step=0
        if not paused:
            order = ["00","01"]
            random.shuffle(order)
            for rid in order:
                agents[rid].step(False)
            step += 1
        # draw
        screen.fill((255,255,255))
        # grid lines
        for i in range(size+1):
            pygame.draw.line(screen, (0,0,0), (0, i*cell), (W, i*cell), 1)
            pygame.draw.line(screen, (0,0,0), (i*cell, 0), (i*cell, H), 1)
        # visited heat
        for y in range(size):
            for x in range(size):
                if world.grid[y][x] == VISITED:
                    pygame.draw.rect(screen, (220,220,220), (x*cell+1, (size-1-y)*cell+1, cell-2, cell-2))
        # clues
        for (cx,cy) in world.clues:
            pygame.draw.rect(screen, (200,200,0), (cx*cell+8, (size-1-cy)*cell+8, cell-16, cell-16))
        # kid
        kx,ky = world.kid
        pygame.draw.circle(screen, (0,150,0), (kx*cell+cell//2, (size-1-ky)*cell+cell//2), cell//3, 0)
        # robots
        for rid, agent in agents.items():
            x,y = agent.pos
            pygame.draw.circle(screen, (0,0,255) if rid=="00" else (255,0,0), (x*cell+cell//2, (size-1-y)*cell+cell//2), cell//3, 0)
        # hud
        txt = font.render(f"step {step}  [space]=pause  [f]=fast  [r]=reset", True, (0,0,0))
        screen.blit(txt, (8,8))
        pygame.display.flip()
        clock.tick(cfg.viewer_fps* (3 if fast else 1))
    pygame.quit()

# ------------------------
# CLI
# ------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["batch","view"], default="batch")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--grid", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="sim_out")
    args = p.parse_args()
    cfg = Config(grid_size=args.grid, seed=args.seed)
    if args.mode == "batch":
        os.makedirs(args.out, exist_ok=True)
        agg = run_batch(cfg, args.episodes, base_seed=args.seed)
        # write summary
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            import json; json.dump(agg["summary"], f, indent=2)
        # write episode csv
        with open(os.path.join(args.out, "episodes.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ep","found","steps","sim_seconds","revisit_00","revisit_01","path_00","path_01","kid_x","kid_y","clues"])
            for i,r in enumerate(agg["results"]):
                w.writerow([i, int(r.found), r.steps, r.sim_seconds,
                            r.revisits_by_robot["00"], r.revisits_by_robot["01"],
                            r.path_length_by_robot["00"], r.path_length_by_robot["01"],
                            r.kid[0], r.kid[1], ";".join(f"{c[0]},{c[1]}" for c in r.clues)])
        print("Wrote:", os.path.join(args.out, "summary.json"))
        print("Wrote:", os.path.join(args.out, "episodes.csv"))
    else:
        run_viewer(cfg, seed=args.seed)

if __name__ == "__main__":
    import os, csv, json
    main()
