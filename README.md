# Mini-Among Us 2D — Real-Time Multiplayer & AI Trajectory Sandbox

A real-time 10 Hz 2D top-down social deduction environment ("Mini-Among Us") with smooth WASD movement (`130 px/s`), true 2D wall-occluded Line-of-Sight shadows, Sabotage Lights, interactive task mini-games, synthesized WebAudio SFX, WebSocket LAN multiplayer, and full 10 Hz keystroke/trajectory logging.

## Architecture & Game Balance

- `engine.py` — **100% pure Python (stdlib only)** 10 Hz (`dt = 0.1s`) 2D fixed-timestep state machine (`MiniAmongUsEngine`) with AABB room/hallway collision & wall sliding, exact Liang-Barsky `has_line_of_sight()`, `get_zone(x, y)`, `130 px/s` speed, `4` assigned tasks per Crewmate (`12` total to win, `5.0s` bot channel or interactive human mini-game), `80 px` kill range, `6.0s` / `8.0s` / `12.0s` kill cooldowns, `25.0s` Sabotage Lights cooldown (`220 px -> 85 px` Crewmate vision vs `260 px` Impostor vision), `2.0s` vent cooldown, `15.0s` emergency cooldown, and natural waypoint bots.
- `server.py` + `templates/index.html` — FastAPI + WebSocket server (`0.0.0.0:5000`) with a 60fps HTML5 `<canvas>` renderer featuring raycasted wall shadows, Among Us bean characters & bone bodies, interactive task popups (Keypad, 1-to-5 Manifolds, Match 3 Wires, 3-Switch Lights Breaker), dramatic Meeting/Ejection banners, WebAudio SFX, and append-only JSONL logging to `data/games_log.jsonl`.
- `export_features.py` — Deterministic pipeline exporting `data/games_log.jsonl` into `data/keystroke_trajectories.csv` (with `lights_out`) and `data/meeting_features.csv`.
- `test_engine.py` — Unit tests, LOS/partial-observability isolation checks, and 20-game headless balance benchmark.

## Controls (Map Phase)

- **Move:** `W` `A` `S` `D` or `Arrow Keys` (or on-screen touch D-pad)
- **Interact:** `E` or `Space` (Report body, fix Lights breaker in Electrical, open task mini-game, or press Cafeteria Emergency Button)
- **Kill (Impostor only):** `Q` (within `80 px` when kill cooldown reaches `0.0s`)
- **Vent (Impostor only):** `V` (within `60 px` of `Medbay <-> Admin` or `Reactor <-> Electrical` grates)
- **Sabotage Lights (Impostor only):** `F` (`25.0s` cooldown; cuts Crewmate vision to `85 px` until repaired in Electrical)

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Tests & 20-Game Balance Check
```bash
python test_engine.py
```

### 3. Host Local Multiplayer Game (Wi-Fi / Hotspot)
```bash
python server.py
```
Open the printed LAN URL (`http://<your-local-ip>:5000`) on teammates' laptops or phones, claim slots `P0`–`P4` (empty slots can be filled with 2D bots), and click **Start Real-Time Game**.

### 4. Export Colab-Ready CSVs
Click **Export CSVs** in the lobby/game-over screen or run:
```bash
python export_features.py
```
Load in Google Colab:
```python
import pandas as pd
from engine import MiniAmongUsEngine, get_zone, has_line_of_sight

traj_df = pd.read_csv("keystroke_trajectories.csv")
meetings_df = pd.read_csv("meeting_features.csv")
```
