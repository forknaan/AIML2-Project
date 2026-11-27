# Mini-Among Us 2D — AI Architecture & Team Implementation Plan

This document specifies the complete hybrid AI architecture, literal from-scratch algorithms, dependency graph (parallel vs. sequential work), frozen module interfaces, and verification checklists for **Mini-Among Us 2D**.

Core game physics, logging, and exports live in [engine.py](engine.py), [server.py](server.py), and [export_features.py](export_features.py). Baseline synthetic data lives in [data/synthetic/](data/synthetic/) (`1,000` headless bot games).

---

## 1. System Overview: Three-Layer Hybrid Agent

Rather than forcing one monolithic neural network to solve geometry, tactics, and social reasoning simultaneously, each AI agent is decomposed into **three clean, auditable algorithms**:

| Layer | Question Answered | Literal Algorithm (From Scratch) | Needs Human Data? |
|---|---|---|---|
| **1. Motion Planning (`motion_planner.py`)** | *How do I move collision-free from $(x, y)$ to target $(g_x, g_y)$?* | **RRT\*** (Optimal Rapidly-exploring Random Tree) + greedy shortcutting + pure-pursuit WASD tracker | **No** (uses `engine.py` map geometry) |
| **2. Tactical RL (`env.py` + `a3c.py`)** | *What high-level action or goal should I pursue right now?* | **Hierarchical A3C** (Asynchronous Advantage Actor-Critic, Mnih et al. 2016) with action masking & `SharedAdam` | **No** (trains online in `engine.py`; optional BC warm-start later) |
| **3. Social Deduction (`social_bayes.py`)** | *During a meeting, who is the Impostor?* | **Hybrid Naive Bayes** + **Structured Bayesian Network** (MLE + Laplace smoothing + exact Variable Elimination) | **Code & tests: No.** **Final parameters: Yes (~200 human games)** |

> **Why RRT\* instead of RL for low-level navigation?**  
> Learning wall-sliding around 6 rooms and 7 narrow corridors via raw WASD RL wastes hundreds of thousands of gradient steps on pure geometry. Separating **RRT\* (low-level collision-free pathing)** from **A3C (high-level tactical goal selection)** lets A3C focus entirely on strategy (task routing, isolation avoidance, stalking, kill timing, venting, sabotaging, and escaping). A `raw_wasd` mode is also kept in `env.py` as an ablation baseline for the final report.

---

## 2. Dependency Graph & Parallel vs. Sequential Execution Plan

### Master Dependency DAG

```text
[DONE] engine.py (10 Hz physics, LOS, collision, bots)
[DONE] export_features.py (23-col CSV extractor)
[DONE] data/synthetic/ (1,000 synthetic bot games)
        │
        ├──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
        ▼                              ▼                              ▼                              ▼
  [TRACK A — PARALLEL]          [TRACK B — PARALLEL]           [TRACK C — PARALLEL]          [TRACK D — PARALLEL]
  M1: motion_planner.py         M2: env.py (scaffold +         M4: social_bayes.py           M5: Human Playtests
  From-scratch RRT* +           obs/mask/rewards; uses         From-scratch Naive Bayes      Collect ~200 real
  WASD tracker + tests          mock/stub planner first)       + Bayes Net + VE + tests      5-human games via
        │                              │                       (verified on synthetic)       server.py + export
        └──────────────┬───────────────┘                              │                              │
                       ▼                                              │                              │
        [SEQUENTIAL — DEPENDS ON M1 + M2]                             │                              │
        M2b: Wire RRT* into env.py                                    │                              │
        (Hierarchical Options Action Space)                           │                              │
                       │                                              │                              │
                       ▼                                              ▼                              ▼
        [SEQUENTIAL — DEPENDS ON M2b]                  [SEQUENTIAL — DEPENDS ON M4 + M5]
        M3: a3c.py + train_a3c.py                      M4b: Fit & Audit Social Bayes on ~200 Human Games
        From-scratch A3C (SharedAdam,                  (Extract real human CPTs, chat_rt_ms / vote_rt_ms,
        N-step returns, workers, self-play)            Naive Bayes vs. Bayes Net GroupKFold CV)
                       │                                              │
                       └──────────────────────┬───────────────────────┘
                                              ▼
                       [SEQUENTIAL — FINAL INTEGRATION (DEPENDS ON M3 + M4b)]
                       M6: Full Hybrid Agent Integration & Evaluation
                       • Plug human-trained Bayes Net into env.py (meeting_mode="bayes_net")
                       • Plug A3C + RRT* + Bayes Net agents into server.py for live human play
                       • Run cross-play tournament & ablations (Raw-WASD vs RRT*+A3C, NB vs BN)
```

### What Can Run in Parallel Right Now (Zero Blockers)

Four independent tracks can be developed **simultaneously**:

- **Track A (Motion Planning):** Build and test `motion_planner.py` (depends only on [engine.py](engine.py) geometry constants & `is_walkable`).
- **Track B (RL Env & A3C Core):** Build `env.py` observation/reward/mask logic and `a3c.py` network + `SharedAdam` + worker loop (verified immediately in `raw_wasd` mode, then wired to Track A's `RRTStarPlanner`).
- **Track C (Social Bayes):** Build `social_bayes.py` (`HybridNaiveBayes`, `AmongUsBayesNet`, exact Variable Elimination, `GroupKFold` evaluator) and verify on `data/synthetic/meeting_features.csv` + existing human rows.
- **Track D (Human Data Collection):** Host playtest sessions via [server.py](server.py) and accumulate ~200 human games in `data/games_log.jsonl`, running `python export_features.py` after each session.

### What Must Be Sequential (Strict Dependencies)

1. **M2b (`env.py` hierarchical options)** waits for **M1 (`motion_planner.py`)** to expose its frozen `plan_path()` + `PathFollower` API.
2. **M3 (Hierarchical A3C training & self-play)** waits for **M2b**.
3. **M4b (Final human CPT fitting & social analysis)** waits for **M4 (`social_bayes.py`)** + **M5 (~200 human games)**.
4. **M6 (End-to-end hybrid agent evaluation & `server.py` live deployment)** waits for **M3** + **M4b**.

---

## 3. Detailed Module Specifications & Literal Algorithms

---

### Module 1: Motion Planning with RRT\* (`motion_planner.py` + `test_motion_planner.py`)

- **Dependencies:** [engine.py](engine.py) (`WALKABLE_RECTS`, `ROOM_AABBS`, `CORRIDOR_AABBS`, `PLAYER_RADIUS`, `is_walkable`, `get_zone`, `VENTS`, `TASK_STATIONS`, `LIGHTS_PANEL_POS`, `EMERGENCY_BUTTON_POS`) + Python stdlib / `numpy` only.
- **Configuration Space ($\mathcal{C}_{\text{free}}$):**
  $$\mathcal{C}_{\text{free}} = \big\{(x, y) \in [0, 1200] \times [0, 800] \;\big|\; \text{is\_walkable}(x, y, r_{\text{plan}})\big\}$$
  where $r_{\text{plan}} = \text{PLAYER\_RADIUS} + \delta_{\text{margin}}$ (`14.0 + 4.0 = 18.0 px`, falling back to `14.0 px` if the start or goal is tight against a wall) so paths keep clearance from doorway corners.

#### Literal RRT\* Algorithm (Karaman & Frazzoli, 2011)

Given `start = (sx, sy)`, `goal = (gx, gy)`, optional dynamic threat circles `threats = [(tx, ty, radius, weight)]`, step size $\eta = 35.0\text{ px}$, iteration budget $N$ (e.g., `400` online, `1500` offline cache), and goal bias $p_{\text{goal}} = 0.15$:

1. **Initialize tree:** $V = \{x_{\text{start}}\}$, $E = \emptyset$, $\text{cost}(x_{\text{start}}) = 0$, $\text{parent}(x_{\text{start}}) = \text{None}$.
2. **For** $i = 1, \dots, N$:
   - **SampleFree():** With probability $p_{\text{goal}}$, set $x_{\text{rand}} = x_{\text{goal}}$. Otherwise sample uniformly inside `WALKABLE_RECTS` weighted by rectangle area (avoids wasting ~65% of samples in non-walkable void space) and accept if `is_walkable(x_rand, r_plan)`.
   - **Nearest($V, x_{\text{rand}}$):** $x_{\text{nearest}} = \arg\min_{v \in V} \|v - x_{\text{rand}}\|_2$.
   - **Steer($x_{\text{nearest}}, x_{\text{rand}}, \eta$):** Move from $x_{\text{nearest}}$ toward $x_{\text{rand}}$ by $\min(\eta, \|x_{\text{rand}} - x_{\text{nearest}}\|_2)$ to obtain $x_{\text{new}}$.
   - **CollisionFree($a, b$):** Sample points along segment $\overline{ab}$ every $\Delta s = 4.0\text{ px}$; valid iff every point satisfies `is_walkable(p, r_plan)` (prevents corner-cutting across AABB walls).
   - **EdgeCost($a, b$):**
     $$c(a, b) = \|b - a\|_2 + \sum_{(t, R, w) \in \text{threats}} w \cdot \text{overlap\_penalty}(\overline{ab}, t, R)$$
     *(Zero threat penalty for standard shortest paths; positive penalty for Impostor escape routes avoiding visible Crewmates or Crewmates routing around a suspected killer.)*
   - **Near($V, x_{\text{new}}$):** Compute shrinking RRT\* rewiring ball radius:
     $$r_n = \min\!\left(\gamma_{\text{RRT}^*} \sqrt{\frac{\log(|V| + 1)}{|V| + 1}},\; r_{\max}\right)$$
     and collect $X_{\text{near}} = \{v \in V \mid \|v - x_{\text{new}}\|_2 \le r_n\}$.
   - **Choose Best Parent:**
     $$x_{\min} = \arg\min_{x \in X_{\text{near}} \cup \{x_{\text{nearest}}\},\; \text{CollisionFree}(x, x_{\text{new}})} \big(\text{cost}(x) + c(x, x_{\text{new}})\big)$$
     Insert $x_{\text{new}}$ with parent $x_{\min}$ and $\text{cost}(x_{\text{new}}) = \text{cost}(x_{\min}) + c(x_{\min}, x_{\text{new}})$.
   - **Rewire Tree:** For every $x_{\text{near}} \in X_{\text{near}}$:
     if $\text{CollisionFree}(x_{\text{new}}, x_{\text{near}})$ and $\text{cost}(x_{\text{new}}) + c(x_{\text{new}}, x_{\text{near}}) < \text{cost}(x_{\text{near}})$:
     reassign $\text{parent}(x_{\text{near}}) \leftarrow x_{\text{new}}$ and propagate the cost delta down its subtree.
3. **Goal Connection:** Connect all $v \in V$ within $r_{\text{goal}}$ of $x_{\text{goal}}$ where `CollisionFree(v, x_goal)`, choose the minimum-cost terminal node, and backtrack parent pointers.
4. **Greedy Line-of-Sight Path Shortcutting:** Iteratively replace sub-paths $(p_i, \dots, p_j)$ with direct segment $(p_i, p_j)$ whenever `CollisionFree(p_i, p_j)` holds and $c(p_i, p_j) \le \sum_{k=i}^{j-1} c(p_k, p_{k+1})$.
5. **Fast A3C Rollout Support (`RRTStarCache` + Online Replanning):**
   - Online `plan_path(start, goal, threats)` runs full RRT\* whenever dynamic threats are present or replanning is triggered (goal change or off-path deviation $> 40\text{ px}$).
   - For high-speed threat-free A3C training, `RRTStarCache` precomputes RRT\* paths between map anchors (room centers, task stations, vents, lights panel, emergency button) once at startup and stitches local start/end segments online.
6. **Pure-Pursuit WASD Controller (`PathFollower`):**
   - Advances waypoint index when within $16\text{ px}$ of the current waypoint, computes $(dx, dy)$ to the active lookahead waypoint, and converts it cleanly to 8-way `(w, a, s, d)` boolean keys matching `engine.step()`.

#### Frozen Interface Contract (`motion_planner.py`)

```python
class RRTStarPlanner:
    def plan_path(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        threats: list[tuple[float, float, float, float]] | None = None,
        max_iter: int = 400,
        seed: int | None = None,
    ) -> list[tuple[float, float]]: ...

class PathFollower:
    def set_goal(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        threats: list[tuple[float, float, float, float]] | None = None,
    ) -> None: ...
    def step_keys(self, pos: tuple[float, float]) -> dict[str, bool]: ...
    def is_arrived(self, pos: tuple[float, float], tol: float = 20.0) -> bool: ...
```

---

### Module 2: Hierarchical RL Environment Wrapper (`env.py` + `test_env.py`)

- **Dependencies:** [engine.py](engine.py) + `numpy` (+ `motion_planner.py` in M2b; works in `raw_wasd` mode immediately in parallel).
- **Two Classes:**
  1. `MiniAmongUsMultiAgentEnv`: steps all 5 slots (`P0`–`P4`); any subset controlled by RL, remaining slots run pluggable policies (heuristic bots, frozen A3C checkpoints, or human/BC policies).
  2. `MiniAmongUsEnv`: single-agent wrapper (`role_mode="crewmate" | "impostor" | "both"`, randomized slot `P0`–`P4` each episode).
- **Strict Partial Observability (`obs` $\in \mathbb{R}^{108}$, documented by `OBS_SPEC`):**
  Built *exclusively* from `engine.get_observation(player_id)`:
  - Self & global state (`15`): normalized $(x, y, v_x, v_y)$, role, alive, doing task, task progress, vision radius, `lights_out`, total tasks `/12`, alive `/5`, `map_time/180`, `round_time/180`, `round/5`.
  - Zone one-hot (`13`): 6 rooms + 7 corridors.
  - Cooldowns & affordances (`12`): emergency CD, meetings called, `can_report`, `can_call_meeting`, `can_start_task`, `near_lights_panel`; Impostor-only (0 for Crewmate): `kill_cd`, `vent_cd`, `sabotage_cd`, `can_kill`, `can_vent`, `can_sabotage`.
  - Objective vectors (`18`): relative $(dx, dy, d)$ + RRT\* initial heading $(u_x, u_y)$ toward nearest assigned task (or nearest vent), Electrical breaker panel, Emergency Button, and remaining personal tasks `/4`.
  - Other 4 players with short-term spatial memory (`4 × 9 = 36`): `is_alive`, `currently_visible`, `rel_dx`, `rel_dy`, `rel_vx`, `rel_vy`, `is_doing_task`, `last_seen_age_norm`, `saw_crime`.
  - Nearest visible body (`4`) + Meeting context (`10`).
- **Action Space Modes (`control_mode`):**
  - **`"hierarchical_rrt"` (Primary Mode — `Discrete(15)` tactical actions + exact binary mask $m_t \in \{0,1\}^{15}$):**
    - `0`: `HOLD_IDLE` (stand still / continue active task channel or lights repair)
    - `1`: `GO_NEAREST_TASK` (RRT\* to nearest incomplete task — or nearest task station to fake as Impostor)
    - `2`: `GO_LIGHTS_PANEL` (RRT\* to Electrical breaker when `lights_out == True`)
    - `3`: `GO_VISIBLE_BODY` (RRT\* to nearest visible dead body)
    - `4`: `GO_EMERGENCY_BUTTON` (RRT\* to Cafeteria button)
    - `5..9`: `PATROL_ROOM_0..4` (RRT\* to `Medbay`, `Admin`, `Storage`, `Reactor`, `Electrical` — also serves as escape room selection)
    - `10`: `STALK_OR_FLEE_PLAYER` (Impostor: RRT\* toward nearest visible Crewmate; Crewmate: threat-penalized RRT\* escape route away from nearest visible player)
    - `11`: `INTERACT` (`E` key: report / fix lights / start task / call emergency)
    - `12`: `KILL` (Impostor only, masked off unless `can_kill_target is not None`)
    - `13`: `VENT` (Impostor only, masked off unless `can_vent == True`)
    - `14`: `SABOTAGE_LIGHTS` (Impostor only, masked off unless `can_sabotage == True`)
  - **`"raw_wasd"` (Ablation Mode — `Discrete(13)`):** `IDLE`, 8-way WASD, `INTERACT`, `KILL`, `VENT`, `SABOTAGE` — enables direct comparison of **Raw-WASD A3C** vs. **Hierarchical A3C + RRT\***.
- **Decision Interval (`frame_skip`):** Default `5` ticks (`0.5s`) in `hierarchical_rrt` mode (`PathFollower` outputs smooth 10 Hz WASD keys across sub-ticks, breaking early on arrival, phase change, kill, or task completion).
- **Meeting Resolution (`meeting_mode`):**
  - `"heuristic"` (default during early Map RL training)
  - `"bayes_net"` (calls `social_bayes.AmongUsBayesNet.predict_meeting_posterior(...)` for Crewmate voting)
- **Auditable Rewards (`RewardConfig` dataclass, logged in `info["reward_terms"]`):**
  - Terminal Win / Loss: `+10.0` / `-10.0`
  - Crewmate: `+2.0` personal task complete, `+0.02/tick` channeling progress, `+1.0` fix lights, `+0.5` report body, `+2.0` Impostor ejected, `-2.0` killed/ejected, `-0.005/step`
  - Impostor: `+3.0` kill, `-1.0` witnessed-kill penalty, `+0.2` sabotage lights, `+0.5` survive meeting, `-3.0` ejected, `-0.003/step`

---

### Module 3: High-Level Tactical RL — From-Scratch **A3C** (`a3c.py` + `test_a3c.py`)

- **Dependencies:** `env.py` + PyTorch (`torch`, `torch.multiprocessing`) — **zero external RL libraries**.
- **Network (`ActorCriticNet`):**
  - Shared trunk: `LayerNorm(108)` → `Linear(108, 128)` → `ReLU` → `Linear(128, 128)` → `ReLU` (optional `GRUCell(128, 128)` flag `use_gru=True` for multi-step fog-of-war memory).
  - Actor head: `Linear(128, |A|)` → raw logits $z_t \in \mathbb{R}^{|\mathcal{A}|}$; Critic head: `Linear(128, 1)` → value $V_\phi(o_t) \in \mathbb{R}$.
  - **Exact Action Masking:**
    $$\tilde{z}_{t,a} = \begin{cases} z_{t,a} & m_{t,a} = 1 \\ -10^9 & m_{t,a} = 0 \end{cases}, \qquad \pi_\theta(a \mid o_t) = \text{softmax}(\tilde{z}_t)_a$$
- **Custom `SharedAdam`:** Subclasses `torch.optim.Optimizer`, pre-allocates and calls `.share_memory_()` on per-parameter `step`, `exp_avg` ($m_t$), and `exp_avg_sq` ($v_t$) tensors for asynchronous updates across CPU worker processes (top-level picklable for Windows `spawn` and Linux/Colab).
- **Asynchronous Worker Loop ($W = 4\text{--}8$ workers, rollout horizon $t_{\max} = 20$):**
  1. Pull global weights: $\theta_{\text{local}} \leftarrow \theta_{\text{global}}$.
  2. Roll out $n \le t_{\max}$ steps storing $(o_k, a_k, m_k, r_k, \log\pi(a_k|o_k), H_k, V(o_k))$.
  3. Bootstrap $R_n = 0$ if terminal else $V_\phi(o_n).\text{detach}()$.
  4. Backward $n$-step return & advantage for $k = n-1, \dots, 0$:
     $$R_k = r_k + \gamma R_{k+1}, \qquad A_k = R_k - V_\phi(o_k)$$
     *(Optional `gae_lambda` toggle for GAE($\lambda$) ablation.)*
  5. Exact A3C objective:
     $$\mathcal{L}_{\text{A3C}} = -\frac{1}{n}\sum_{k=0}^{n-1}\log\pi_\theta(a_k \mid o_k)\,\text{sg}(A_k) + \frac{c_v}{2n}\sum_{k=0}^{n-1}\big(R_k - V_\phi(o_k)\big)^2 - \frac{c_e}{n}\sum_{k=0}^{n-1}H\big(\pi_\theta(\cdot \mid o_k)\big)$$
     ($\gamma = 0.99$, $c_v = 0.5$, $c_e = 0.02 \to 0.005$, gradient norm clip $\|\nabla\|_2 \le 40.0$.)
  6. Compute `loss.backward()` locally, copy local gradients onto `global_net`, and step `SharedAdam`.
- **Training Curriculum:**
  1. **Phase 1:** Train Crewmate A3C vs. heuristic Impostor & Impostor A3C vs. heuristic Crewmates.
  2. **Phase 2 (Self-Play):** Alternating best-response — freeze one role's A3C checkpoint as opponent while updating the other.
  3. **Phase 3 (Ablation):** Compare `hierarchical_rrt` A3C vs. `raw_wasd` A3C learning curves and win rates.

---

### Module 4: Social Deduction — From-Scratch **Naive Bayes + Bayesian Network** (`social_bayes.py` + `test_social_bayes.py`)

- **Dependencies:** `numpy` + Python stdlib (`csv`, `json`, `math`) — **zero `sklearn` / `pgmpy` black boxes**.
- **Strict Leakage Policy:** Drops `voted_for_impostor` unconditionally; default `strict_partial_obs=True` also excludes omniscient `min_dist_to_body_last_seen` (kept only for optional post-game offline audit).

#### Model 4A: `HybridNaiveBayes`

- **Discrete features** (`reported_body`, `chat_type`, `logic_impossible_travel`, `logic_witness_contradiction`): Categorical MLE with Laplace smoothing $\alpha$:
  $$P(X_j = v \mid I = c) = \frac{N_{c,j,v} + \alpha}{N_c + \alpha |V_j|}$$
- **Continuous features** (`time_alone_ratio`, `avg_speed`, `idle_ratio`, `direction_change_rate`, `unique_witnesses_count`, `unique_rooms_visited`, `vouches_received`, `accusations_received`, `chat_rt_ms`, `vote_rt_ms`): Gaussian MLE with variance floor $\epsilon = 10^{-6}$:
  $$\log p(x_j \mid I = c) = -\tfrac{1}{2}\log\big(2\pi(\sigma_{c,j}^2 + \epsilon)\big) - \frac{(x_j - \mu_{c,j})^2}{2(\sigma_{c,j}^2 + \epsilon)}$$
- **Exact Missing-Value Marginalization:** `NaN` values (`chat_rt_ms`/`vote_rt_ms` on bots) contribute `0` to log-likelihood rather than distorting means/variances.
- **Meeting-Level 1-Impostor Normalization:** Across alive candidates $\mathcal{C} = \mathcal{A} \setminus \{\text{self}\}$:
  $$P(\text{Impostor} = p \mid \mathbf{E}_{\mathcal{A}}) = \frac{\exp(\ell_p)}{\sum_{q \in \mathcal{C}} \exp(\ell_q)}$$

#### Model 4B: `AmongUsBayesNet` (Causal DAG + Exact Variable Elimination)

```text
               ┌──► Isolation ──► WitnessCount ──► VouchesReceived
               │                        │
               ├──► MovementProfile     ▼
               │               LogicWitnessContradiction ──► AccusationsReceived ──► VoteRT
Impostor (I) ──┼──► ChatType ──────────►┘                                              ▲
               │       │  └──► LogicImpossibleTravel                                   │
               │       └─────► ChatRT                                                  │
               ├──► ReportedBody                                                       │
               └───────────────────────────────────────────────────────────────────────┘
```

- **Leak-Free Discretization:** Continuous features binned into interpretable states (`Low/Med/High`, `Fast/Med/Slow`) using **training-fold quantiles only**.
- **CPT Parameter Learning (MLE + Dirichlet/Laplace $\alpha$):**
  $$\hat{P}(X_i = x_i \mid \text{Pa}(X_i) = \mathbf{u}) = \frac{N(x_i, \mathbf{u}) + \alpha}{N(\mathbf{u}) + \alpha |X_i|}$$
  Every CPT is exportable/printable as a JSON or Markdown table for human auditing.
- **Exact Inference via Variable Elimination (`Factor` class in pure NumPy):**
  1. Restrict CPT factors to observed evidence $X_j = e_j$.
  2. Multiply factors and sum out unobserved/`NaN` variables to obtain exact $P(I = c, \mathbf{e}_{\text{obs}})$.
  3. Compute both individual $P(I = 1 \mid \mathbf{e})$ and meeting-level 1-Impostor posterior across $\mathcal{C}$.
- **Explainability (`explain_prediction(row)`):** Outputs per-node CPT probabilities and log-Bayes-factors showing which behaviors, logic contradictions, and reaction times drove suspicion.

#### Frozen Interface Contract (`social_bayes.py`)

```python
class AmongUsBayesNet:
    def fit(self, rows: list[dict], alpha: float = 1.0) -> "AmongUsBayesNet": ...
    def predict_proba_row(self, row: dict) -> float: ...
    def predict_meeting_posterior(
        self, meeting_rows: list[dict], self_player: str | None = None
    ) -> dict[str, float]: ...
    def explain_prediction(self, row: dict) -> list[dict]: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "AmongUsBayesNet": ...
```

---

### Module 5: Human Data Collection (~200 Games) — Parallel Track

- Host multiplayer sessions with `python server.py` over local Wi-Fi/hotspot; filter out pre-rebalance test games (e.g., `20261124_175008_40bf`) during training.
- Run `python export_features.py` after each session to update `data/keystroke_trajectories.csv` and `data/meeting_features.csv`.
- Once ~200 games are collected, fit and save the human social model:
  ```bash
  python social_bayes.py --input data/meeting_features.csv --human-only --save models/human_bayes_net.json
  ```

---

### Module 6: Integration, Ablations & Live Deployment (Final Sequential Phase)

1. **Wire `models/human_bayes_net.json` into `env.py` (`meeting_mode="bayes_net"`):** Crewmate agents vote using the human-calibrated Bayesian Network posterior.
2. **Wire AI agents into [server.py](server.py):** Allow empty lobby slots to select `Heuristic Bot` vs. `A3C + RRT* + Bayes Net AI` so humans can play live against the trained system.
3. **Final Evaluation Matrix (`evaluate_agents.py` / Colab notebook):**
   - **Motion Planning Audit:** RRT\* path optimality ratio vs. shortest doorway path, collision rate (`0%` target), threat-avoidance clearance, replanning latency (ms).
   - **RL Ablations:** Win rate, tasks/game, kills/game, and learning curves for `Heuristic Bot` vs. `Raw-WASD A3C` vs. `Hierarchical A3C + RRT*` vs. `Self-Play A3C + RRT*`.
   - **Social Deduction Audit (on ~200 human games, `GroupKFold` by `game_id`):** `Logic Rules` vs. `HybridNaiveBayes` vs. `AmongUsBayesNet` on **ROC-AUC, PR-AUC, F1, Brier Score, Log-Loss, and Top-1 Meeting Impostor Identification Accuracy**, plus printed human CPT tables (`ChatRT` and `VoteRT` by role and speech act).

---

## 4. Per-Module Audit & Verification Checklists

| Module | Test File | Mandatory Passing Checks |
|---|---|---|
| **M1: RRT\*** | `test_motion_planner.py` | 100% of sampled path segments collision-free (`is_walkable`), connects every room pair across corridors, cost improves with rewiring (`RRT* <= RRT`), threat-avoidance routes around hazard zones, `PathFollower` reaches target inside `engine.py`. |
| **M2: RL Env** | `test_env.py` | Zero partial-observability leakage (mutating off-vision players/bodies changes `obs` by `0.0`), legal masks verified across all states, both `hierarchical_rrt` and `raw_wasd` modes complete episodes cleanly, headless throughput benchmarked. |
| **M3: A3C** | `test_a3c.py` | Masked softmax gives exact `0.0` probability on illegal actions, $n$-step return/advantage numerical unit test, multi-worker `SharedAdam` weight-sync test on Windows `spawn`, smoke training run logs finite losses & improving returns. |
| **M4: Social Bayes** | `test_social_bayes.py` | Every CPT sums to `1.0` ($\pm 10^{-12}$), Variable Elimination matches brute-force joint-table enumeration to $< 10^{-12}$ (including missing `NaN` nodes), zero `game_id` leakage in `GroupKFold`, `--human-only` CLI verified. |
