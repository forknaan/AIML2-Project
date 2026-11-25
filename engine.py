"""Headless pure-Python 10 Hz 2D engine for Mini-Among Us (stdlib only).

Fixed-timestep continuous 2D top-down simulation (TICK_RATE = 10 Hz, dt = 0.1s)
with AABB room + hallway collision, diagonal wall sliding, fog-of-war partial
observability, real-time cooldowns, structured Quick Chat meetings, and
complete 10 Hz keystroke/trajectory logging.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import heapq
import math
import random
from typing import Any
import uuid

PLAYERS: tuple[str, ...] = ("P0", "P1", "P2", "P3", "P4")
ROOMS: tuple[str, ...] = (
    "Cafeteria",
    "Medbay",
    "Admin",
    "Storage",
    "Reactor",
    "Electrical",
)
TASK_ROOMS: tuple[str, ...] = (
    "Medbay",
    "Admin",
    "Storage",
    "Reactor",
    "Electrical",
)

TICK_RATE: int = 10
DEFAULT_DT: float = 0.1

WORLD_WIDTH: float = 1200.0
WORLD_HEIGHT: float = 800.0
PLAYER_RADIUS: float = 14.0
PLAYER_SPEED: float = 130.0

KILL_RANGE: float = 80.0
INTERACT_RANGE: float = 70.0
VENT_RANGE: float = 60.0

INITIAL_KILL_COOLDOWN: float = 6.0
POST_MEETING_KILL_COOLDOWN: float = 8.0
POST_KILL_COOLDOWN: float = 12.0
KILL_COOLDOWN_START: float = INITIAL_KILL_COOLDOWN
KILL_COOLDOWN_AFTER_KILL: float = POST_KILL_COOLDOWN
EMERGENCY_COOLDOWN_START: float = 15.0
VENT_COOLDOWN: float = 2.0
TASK_DURATION: float = 5.0
SABOTAGE_COOLDOWN: float = 25.0
SABOTAGE_INITIAL_COOLDOWN: float = 8.0
BOT_FIX_LIGHTS_DURATION: float = 2.0

CREWMATE_VISION: float = 220.0
CREWMATE_LIGHTS_OUT_VISION: float = 85.0
IMPOSTOR_VISION: float = 260.0

TASKS_PER_CREWMATE: int = 4
TASK_WIN_THRESHOLD: int = 12
TASKS_WIN_THRESHOLD: int = TASK_WIN_THRESHOLD
MAX_MAP_TIME: float = 180.0
MAP_VERSION: str = "6room_2d_v2"

# Room AABBs: (x1, y1, x2, y2) on 1200x800 world
ROOM_AABBS: dict[str, tuple[float, float, float, float]] = {
    "Cafeteria": (450.0, 55.0, 750.0, 325.0),
    "Medbay": (80.0, 70.0, 340.0, 310.0),
    "Admin": (860.0, 70.0, 1120.0, 310.0),
    "Reactor": (80.0, 490.0, 340.0, 730.0),
    "Storage": (460.0, 480.0, 740.0, 740.0),
    "Electrical": (860.0, 490.0, 1120.0, 730.0),
}

ROOM_CENTERS: dict[str, tuple[float, float]] = {
    r: ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
    for r, (x1, y1, x2, y2) in ROOM_AABBS.items()
}

# Walkable hallway AABBs overlap 40px (>= 2*PLAYER_RADIUS) into both rooms
CORRIDOR_AABBS: dict[str, tuple[float, float, float, float]] = {
    "Corridor_Cafeteria_Medbay": (300.0, 150.0, 490.0, 230.0),
    "Corridor_Admin_Cafeteria": (710.0, 150.0, 900.0, 230.0),
    "Corridor_Cafeteria_Storage": (560.0, 285.0, 640.0, 520.0),
    "Corridor_Medbay_Reactor": (170.0, 270.0, 250.0, 530.0),
    "Corridor_Admin_Electrical": (950.0, 270.0, 1030.0, 530.0),
    "Corridor_Reactor_Storage": (300.0, 570.0, 500.0, 650.0),
    "Corridor_Electrical_Storage": (700.0, 570.0, 900.0, 650.0),
}

# Visual corridor spans (between room edges) for crisp canvas floor/wall rendering
CORRIDOR_VISUAL_RECTS: dict[str, tuple[float, float, float, float]] = {
    "Corridor_Cafeteria_Medbay": (340.0, 150.0, 450.0, 230.0),
    "Corridor_Admin_Cafeteria": (750.0, 150.0, 860.0, 230.0),
    "Corridor_Cafeteria_Storage": (560.0, 325.0, 640.0, 480.0),
    "Corridor_Medbay_Reactor": (170.0, 310.0, 250.0, 490.0),
    "Corridor_Admin_Electrical": (950.0, 310.0, 1030.0, 490.0),
    "Corridor_Reactor_Storage": (340.0, 570.0, 460.0, 650.0),
    "Corridor_Electrical_Storage": (740.0, 570.0, 860.0, 650.0),
}

CORRIDOR_LINKS: dict[str, tuple[str, str]] = {
    "Corridor_Cafeteria_Medbay": ("Medbay", "Cafeteria"),
    "Corridor_Admin_Cafeteria": ("Cafeteria", "Admin"),
    "Corridor_Cafeteria_Storage": ("Cafeteria", "Storage"),
    "Corridor_Medbay_Reactor": ("Medbay", "Reactor"),
    "Corridor_Admin_Electrical": ("Admin", "Electrical"),
    "Corridor_Reactor_Storage": ("Reactor", "Storage"),
    "Corridor_Electrical_Storage": ("Storage", "Electrical"),
}

WALKABLE_RECTS: tuple[tuple[float, float, float, float], ...] = tuple(
    list(ROOM_AABBS.values()) + list(CORRIDOR_AABBS.values())
)
OPEN_SPACE_RECTS: tuple[tuple[float, float, float, float], ...] = tuple(
    list(ROOM_AABBS.values()) + list(CORRIDOR_VISUAL_RECTS.values())
)

NORMAL_EDGES: dict[str, tuple[str, ...]] = {
    "Cafeteria": ("Medbay", "Admin", "Storage"),
    "Medbay": ("Cafeteria", "Reactor"),
    "Admin": ("Cafeteria", "Electrical"),
    "Storage": ("Cafeteria", "Electrical", "Reactor"),
    "Reactor": ("Medbay", "Storage"),
    "Electrical": ("Admin", "Storage"),
}

# Doorway portal coordinates connecting adjacent rooms cleanly down corridor centerlines
DOORWAY_PORTALS: dict[tuple[str, str], tuple[float, float]] = {
    ("Medbay", "Cafeteria"): (340.0, 190.0),
    ("Cafeteria", "Medbay"): (450.0, 190.0),
    ("Cafeteria", "Admin"): (750.0, 190.0),
    ("Admin", "Cafeteria"): (860.0, 190.0),
    ("Cafeteria", "Storage"): (600.0, 325.0),
    ("Storage", "Cafeteria"): (600.0, 480.0),
    ("Medbay", "Reactor"): (210.0, 310.0),
    ("Reactor", "Medbay"): (210.0, 490.0),
    ("Admin", "Electrical"): (990.0, 310.0),
    ("Electrical", "Admin"): (990.0, 490.0),
    ("Reactor", "Storage"): (340.0, 610.0),
    ("Storage", "Reactor"): (460.0, 610.0),
    ("Storage", "Electrical"): (740.0, 610.0),
    ("Electrical", "Storage"): (860.0, 610.0),
}

EMERGENCY_BUTTON_POS: tuple[float, float] = (600.0, 190.0)
LIGHTS_PANEL_POS: tuple[float, float] = (915.0, 540.0)

SPAWN_POINTS: dict[str, tuple[float, float]] = {
    "P0": (600.0, 132.0),
    "P1": (655.0, 172.0),
    "P2": (634.0, 237.0),
    "P3": (566.0, 237.0),
    "P4": (545.0, 172.0),
}

TASK_STATIONS: tuple[dict[str, Any], ...] = (
    {"station_id": "Medbay_Scan", "room": "Medbay", "label": "Submit Scan", "x": 135.0, "y": 120.0},
    {"station_id": "Medbay_Samples", "room": "Medbay", "label": "Inspect Samples", "x": 285.0, "y": 120.0},
    {"station_id": "Admin_Swipe", "room": "Admin", "label": "Swipe Card", "x": 915.0, "y": 120.0},
    {"station_id": "Admin_Upload", "room": "Admin", "label": "Upload Data", "x": 1065.0, "y": 120.0},
    {"station_id": "Storage_Fuel", "room": "Storage", "label": "Fuel Engines", "x": 515.0, "y": 695.0},
    {"station_id": "Storage_Trash", "room": "Storage", "label": "Empty Chute", "x": 685.0, "y": 695.0},
    {"station_id": "Reactor_Manifolds", "room": "Reactor", "label": "Unlock Manifolds", "x": 135.0, "y": 540.0},
    {"station_id": "Reactor_Core", "room": "Reactor", "label": "Start Reactor", "x": 135.0, "y": 680.0},
    {"station_id": "Electrical_Wires", "room": "Electrical", "label": "Fix Wiring", "x": 1065.0, "y": 540.0},
    {"station_id": "Electrical_Calibrate", "room": "Electrical", "label": "Calibrate Distributor", "x": 1065.0, "y": 680.0},
)
STATION_BY_ID: dict[str, dict[str, Any]] = {
    st["station_id"]: st for st in TASK_STATIONS
}
STATIONS_BY_ROOM: dict[str, list[dict[str, Any]]] = {
    r: [st for st in TASK_STATIONS if st["room"] == r] for r in TASK_ROOMS
}

VENTS: tuple[dict[str, Any], ...] = (
    {"id": "Vent_Medbay", "room": "Medbay", "x": 135.0, "y": 265.0, "target_id": "Vent_Admin"},
    {"id": "Vent_Admin", "room": "Admin", "x": 1065.0, "y": 265.0, "target_id": "Vent_Medbay"},
    {"id": "Vent_Reactor", "room": "Reactor", "x": 285.0, "y": 685.0, "target_id": "Vent_Electrical"},
    {"id": "Vent_Electrical", "room": "Electrical", "x": 915.0, "y": 685.0, "target_id": "Vent_Reactor"},
)
VENT_BY_ID: dict[str, dict[str, Any]] = {v["id"]: v for v in VENTS}

ACCUSE_REASONS: tuple[str, ...] = (
    "SAW_KILL",
    "SAW_VENT",
    "BAD_ALIBI",
    "SUSPICIOUS",
)
CHAT_TYPES: tuple[str, ...] = ("CLAIM_LOC", "VOUCH", "ACCUSE", "PASS")
TICK_ACTIONS: tuple[str, ...] = (
    "NONE",
    "DO_TASK",
    "KILL",
    "VENT",
    "REPORT",
    "MEETING",
    "SABOTAGE",
    "FIX_LIGHTS",
)


def _clip_segment_rect(
    x1: float,
    y1: float,
    dx: float,
    dy: float,
    rx1: float,
    ry1: float,
    rx2: float,
    ry2: float,
) -> tuple[float, float] | None:
    """Liang-Barsky clip of parametric segment `P(t) = (x1+t*dx, y1+t*dy)`, `t in [0,1]`."""
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x1 - rx1),
        (dx, rx2 - x1),
        (-dy, y1 - ry1),
        (dy, ry2 - y1),
    ):
        if p == 0.0:
            if q < -1e-9:
                return None
        else:
            r = q / p
            if p < 0.0:
                if r > t1:
                    return None
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return None
                if r < t1:
                    t1 = r
    if t0 <= t1 + 1e-9:
        return (t0, t1)
    return None


def has_line_of_sight(x1: float, y1: float, x2: float, y2: float) -> bool:
    """Return True if segment `(x1, y1) -> (x2, y2)` stays strictly inside open rooms/corridors."""
    if x1 == x2 and y1 == y2:
        return True

    # Fast path: both points inside the same convex room/corridor rectangle
    for rx1, ry1, rx2, ry2 in OPEN_SPACE_RECTS:
        if rx1 <= x1 <= rx2 and ry1 <= y1 <= ry2 and rx1 <= x2 <= rx2 and ry1 <= y2 <= ry2:
            return True

    dx = x2 - x1
    dy = y2 - y1
    seg_min_x = x1 if x1 < x2 else x2
    seg_max_x = x2 if x1 < x2 else x1
    seg_min_y = y1 if y1 < y2 else y2
    seg_max_y = y2 if y1 < y2 else y1

    intervals: list[tuple[float, float]] = []
    for rx1, ry1, rx2, ry2 in OPEN_SPACE_RECTS:
        if rx2 < seg_min_x or rx1 > seg_max_x or ry2 < seg_min_y or ry1 > seg_max_y:
            continue
        clipped = _clip_segment_rect(x1, y1, dx, dy, rx1, ry1, rx2, ry2)
        if clipped is not None:
            intervals.append(clipped)

    if not intervals:
        return False

    intervals.sort()
    if intervals[0][0] > 1e-5:
        return False
    covered = intervals[0][1]
    for a, b in intervals[1:]:
        if a > covered + 1e-5:
            return False
        if b > covered:
            covered = b
        if covered >= 1.0 - 1e-5:
            return True
    return covered >= 1.0 - 1e-5


def is_walkable(x: float, y: float, radius: float = PLAYER_RADIUS) -> bool:
    """Return True if circle `(x, y, radius)` stays inside at least one walkable AABB."""
    for x1, y1, x2, y2 in WALKABLE_RECTS:
        if x - radius >= x1 and x + radius <= x2 and y - radius >= y1 and y + radius <= y2:
            return True
    return False


def move_with_sliding(
    x: float,
    y: float,
    dx: float,
    dy: float,
    radius: float = PLAYER_RADIUS,
) -> tuple[float, float]:
    """Advance `(x, y)` by `(dx, dy)` with smooth AABB wall sliding."""
    if dx == 0.0 and dy == 0.0:
        return (x, y)
    if is_walkable(x + dx, y + dy, radius):
        return (x + dx, y + dy)

    nx = x
    if dx != 0.0:
        if is_walkable(x + dx, y, radius):
            nx = x + dx
        else:
            lo, hi = 0.0, 1.0
            for _ in range(6):
                mid = 0.5 * (lo + hi)
                if is_walkable(x + dx * mid, y, radius):
                    lo = mid
                else:
                    hi = mid
            nx = x + dx * lo

    ny = y
    if dy != 0.0:
        if is_walkable(nx, y + dy, radius):
            ny = y + dy
        else:
            lo, hi = 0.0, 1.0
            for _ in range(6):
                mid = 0.5 * (lo + hi)
                if is_walkable(nx, y + dy * mid, radius):
                    lo = mid
                else:
                    hi = mid
            ny = y + dy * lo

    return (nx, ny)


def get_zone(x: float, y: float) -> str:
    """Map continuous `(x, y)` coordinates to a room name or `Corridor_A_B`."""
    for room, (x1, y1, x2, y2) in ROOM_AABBS.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return room
    for corr, (x1, y1, x2, y2) in CORRIDOR_AABBS.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return corr
    best_room = "Cafeteria"
    best_d2 = float("inf")
    for room, (cx, cy) in ROOM_CENTERS.items():
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_room = room
    return best_room


def _build_portal_graph() -> tuple[
    list[tuple[str, str]],
    dict[tuple[str, str], list[tuple[tuple[str, str], float]]],
    dict[str, list[tuple[str, str]]],
]:
    """Precompute doorway portal graph for fast shortest-walkable-distance queries."""
    nodes = list(DOORWAY_PORTALS.keys())
    adj: dict[tuple[str, str], list[tuple[tuple[str, str], float]]] = {n: [] for n in nodes}
    portals_in_room: dict[str, list[tuple[str, str]]] = {r: [] for r in ROOMS}
    for n in nodes:
        portals_in_room[n[0]].append(n)

    # Corridor transitions
    for r1, r2 in nodes:
        p1 = DOORWAY_PORTALS[(r1, r2)]
        p2 = DOORWAY_PORTALS[(r2, r1)]
        d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        adj[(r1, r2)].append(((r2, r1), d))

    # Intra-room transitions (rooms are convex rectangles)
    for room, plist in portals_in_room.items():
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                a, b = plist[i], plist[j]
                pa, pb = DOORWAY_PORTALS[a], DOORWAY_PORTALS[b]
                d = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
                adj[a].append((b, d))
                adj[b].append((a, d))

    return nodes, adj, portals_in_room


_PORTAL_NODES, _PORTAL_ADJ, _PORTALS_IN_ROOM = _build_portal_graph()


def min_walk_dist_to_room(x: float, y: float, target_room: str) -> float:
    """Shortest non-vent walkable Euclidean distance from `(x, y)` to `target_room`."""
    if target_room not in ROOM_AABBS:
        return float("inf")
    cur_zone = get_zone(x, y)
    if cur_zone == target_room:
        return 0.0

    # Dijkstra from virtual start node over doorway portals
    dist: dict[tuple[str, str], float] = {}
    pq: list[tuple[float, tuple[str, str]]] = []

    if cur_zone in ROOM_AABBS:
        start_portals = _PORTALS_IN_ROOM[cur_zone]
    elif cur_zone in CORRIDOR_LINKS:
        r1, r2 = CORRIDOR_LINKS[cur_zone]
        start_portals = [(r1, r2), (r2, r1)]
    else:
        start_portals = _PORTAL_NODES

    for node in start_portals:
        px, py = DOORWAY_PORTALS[node]
        d0 = math.hypot(x - px, y - py)
        dist[node] = d0
        heapq.heappush(pq, (d0, node))

    while pq:
        d_cur, node = heapq.heappop(pq)
        if node[0] == target_room:
            return d_cur
        if d_cur > dist.get(node, float("inf")):
            continue
        for nxt, w in _PORTAL_ADJ[node]:
            nd = d_cur + w
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                heapq.heappush(pq, (nd, nxt))

    return float("inf")


def _compute_room_bfs_parents() -> dict[str, dict[str, str | None]]:
    """Precompute next-room lookup over NORMAL_EDGES for smooth bot routing."""
    next_hop: dict[str, dict[str, str | None]] = {}
    for target in ROOMS:
        parent: dict[str, str | None] = {target: None}
        q: deque[str] = deque([target])
        while q:
            cur = q.popleft()
            for nb in NORMAL_EDGES[cur]:
                if nb not in parent:
                    parent[nb] = cur
                    q.append(nb)
        next_hop[target] = parent
    return next_hop


_ROOM_NEXT_HOP = _compute_room_bfs_parents()


def format_time(seconds: float) -> str:
    """Format elapsed seconds as M:SS (e.g., 18.4 -> '0:18')."""
    total = max(0, int(seconds))
    mins = total // 60
    secs = total % 60
    return f"{mins}:{secs:02d}"


def get_round_time_windows(round_start_t: float, meeting_t: float) -> list[str]:
    """Split [round_start_t, meeting_t] into 10-second 'M:SS-M:SS' windows."""
    s0 = max(0, int(round_start_t))
    s1 = max(s0 + 1, int(math.ceil(meeting_t)))
    windows: list[str] = []
    cur = s0
    while cur < s1:
        nxt = min(cur + 10, s1)
        windows.append(f"{format_time(cur)}-{format_time(nxt)}")
        cur = nxt
    return windows


def parse_time_window(win_str: str) -> tuple[float, float] | None:
    """Parse 'M:SS-M:SS' into `(start_sec, end_sec)` or None if invalid."""
    if not isinstance(win_str, str) or "-" not in win_str:
        return None
    parts = win_str.split("-", 1)
    parsed: list[float] = []
    for part in parts:
        if ":" not in part:
            return None
        m_str, s_str = part.split(":", 1)
        try:
            m_val = int(m_str)
            s_val = int(s_str)
        except ValueError:
            return None
        if m_val < 0 or s_val < 0 or s_val >= 60:
            return None
        parsed.append(float(m_val * 60 + s_val))
    if parsed[1] <= parsed[0]:
        return None
    return (parsed[0], parsed[1])


def format_speech_act(chat_type: str, args: dict[str, Any] | None = None) -> str:
    """Render a structured Quick Chat speech act into readable English."""
    payload = args or {}
    if chat_type == "CLAIM_LOC":
        return f"I was in {payload['room']} during {payload['time_window']}."
    if chat_type == "VOUCH":
        return f"I saw {payload['target']} in {payload['room']}."
    if chat_type == "ACCUSE":
        return f"I accuse {payload['target']} ({payload['reason']})."
    if chat_type == "PASS":
        return "No comment."
    raise ValueError(f"Unknown chat_type: {chat_type}")


def _empty_input() -> dict[str, bool]:
    return {
        "w": False,
        "a": False,
        "s": False,
        "d": False,
        "interact": False,
        "kill": False,
        "vent": False,
        "sabotage": False,
    }


class MiniAmongUsEngine:
    """Deterministic 10 Hz 2D real-time state machine for Mini-Among Us."""

    def __init__(
        self,
        seed: int | None = None,
        is_human: dict[str, bool] | None = None,
        game_id: str | None = None,
        record_ticks: bool = True,
    ) -> None:
        self.rng: random.Random = random.Random(seed)
        self.record_ticks: bool = record_ticks
        self.game_id: str = ""
        self.timestamp_utc: str = ""
        self.is_human: dict[str, bool] = {}
        self.roles: dict[str, str] = {}
        self.impostor_id: str = ""
        self.assigned_tasks: dict[str, list[dict[str, Any]]] = {}
        self.incomplete_tasks: dict[str, set[str]] = {}

        self.positions: dict[str, tuple[float, float]] = {}
        self.velocities: dict[str, tuple[float, float]] = {}
        self.alive_set: set[str] = set()
        self.bodies: list[dict[str, Any]] = []
        self.meetings_called: dict[str, int] = {}
        self.tasks_completed_total: int = 0

        self.kill_cooldown: float = INITIAL_KILL_COOLDOWN
        self.vent_cooldown: float = 0.0
        self.emergency_cooldown: float = EMERGENCY_COOLDOWN_START
        self.lights_out: bool = False
        self.sabotage_cooldown: float = SABOTAGE_INITIAL_COOLDOWN
        self.fixing_lights_progress: dict[str, float] = {}

        self.is_doing_task: dict[str, bool] = {}
        self.active_station: dict[str, str | None] = {}
        self.task_progress: dict[str, float] = {}
        self.last_room: dict[str, str] = {}
        self.prev_visible: dict[str, set[str]] = {}
        self.round_sightings: dict[str, list[dict[str, Any]]] = {}

        self.tick_count: int = 0
        self.map_time: float = 0.0
        self.round_num: int = 1
        self.round_start_time: float = 0.0
        self.round_time: float = 0.0
        self.phase: str = "MAP"
        self.winner: str | None = None
        self.win_reason: str | None = None

        self.ticks_log: list[dict[str, Any]] = []
        self.events_log: list[dict[str, Any]] = []
        self.meetings_log: list[dict[str, Any]] = []

        self.pending_votes: dict[str, dict[str, Any]] = {}
        self.current_meeting: dict[str, Any] | None = None
        self.speaking_order: list[str] = []
        self.speaker_idx: int = 0
        self.last_meeting_summary: dict[str, Any] | None = None

        self.reset(seed=seed, is_human=is_human, game_id=game_id, record_ticks=record_ticks)

    def reset(
        self,
        seed: int | None = None,
        is_human: dict[str, bool] | None = None,
        game_id: str | None = None,
        record_ticks: bool | None = None,
    ) -> None:
        """Reset engine to Cafeteria spawn at t = 0.0s."""
        if seed is not None:
            self.rng = random.Random(seed)
        if record_ticks is not None:
            self.record_ticks = record_ticks

        now = datetime.now(timezone.utc)
        self.timestamp_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.game_id = game_id or f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"

        human_flags = is_human or {}
        self.is_human = {p: bool(human_flags.get(p, False)) for p in PLAYERS}

        impostor = self.rng.choice(PLAYERS)
        self.impostor_id = impostor
        self.roles = {
            p: ("IMPOSTOR" if p == impostor else "CREWMATE") for p in PLAYERS
        }

        self.assigned_tasks = {}
        self.incomplete_tasks = {}
        for p in PLAYERS:
            if self.roles[p] == "CREWMATE":
                chosen_rooms = self.rng.sample(TASK_ROOMS, TASKS_PER_CREWMATE)
                stations = [self.rng.choice(STATIONS_BY_ROOM[r]) for r in chosen_rooms]
                self.assigned_tasks[p] = [
                    {
                        "station_id": st["station_id"],
                        "room": st["room"],
                        "label": st["label"],
                        "x": st["x"],
                        "y": st["y"],
                    }
                    for st in stations
                ]
                self.incomplete_tasks[p] = {st["station_id"] for st in stations}
            else:
                self.assigned_tasks[p] = []
                self.incomplete_tasks[p] = set()

        self.positions = {p: SPAWN_POINTS[p] for p in PLAYERS}
        self.velocities = {p: (0.0, 0.0) for p in PLAYERS}
        self.alive_set = set(PLAYERS)
        self.bodies = []
        self.meetings_called = {p: 0 for p in PLAYERS}
        self.tasks_completed_total = 0

        self.kill_cooldown = INITIAL_KILL_COOLDOWN
        self.vent_cooldown = 0.0
        self.emergency_cooldown = EMERGENCY_COOLDOWN_START
        self.lights_out = False
        self.sabotage_cooldown = SABOTAGE_INITIAL_COOLDOWN
        self.fixing_lights_progress = {p: 0.0 for p in PLAYERS}

        self.is_doing_task = {p: False for p in PLAYERS}
        self.active_station = {p: None for p in PLAYERS}
        self.task_progress = {p: 0.0 for p in PLAYERS}
        self.last_room = {p: "Cafeteria" for p in PLAYERS}
        self.prev_visible = {p: set() for p in PLAYERS}
        self.round_sightings = {p: [] for p in PLAYERS}

        self.tick_count = 0
        self.map_time = 0.0
        self.round_num = 1
        self.round_start_time = 0.0
        self.round_time = 0.0
        self.phase = "MAP"
        self.winner = None
        self.win_reason = None

        self.ticks_log = []
        self.events_log = []
        self.meetings_log = []

        self.pending_votes = {}
        self.current_meeting = None
        self.speaking_order = []
        self.speaker_idx = 0
        self.last_meeting_summary = None

    def clone(self, seed: int | None = None) -> MiniAmongUsEngine:
        """Fast manual deep clone for headless RL / Game Theory rollouts."""
        other = object.__new__(MiniAmongUsEngine)
        if seed is not None:
            other.rng = random.Random(seed)
        else:
            r_copy = random.Random()
            r_copy.setstate(self.rng.getstate())
            other.rng = r_copy

        other.record_ticks = self.record_ticks
        other.game_id = self.game_id
        other.timestamp_utc = self.timestamp_utc
        other.is_human = dict(self.is_human)
        other.roles = dict(self.roles)
        other.impostor_id = self.impostor_id
        other.assigned_tasks = {
            p: [dict(st) for st in lst] for p, lst in self.assigned_tasks.items()
        }
        other.incomplete_tasks = {p: set(s) for p, s in self.incomplete_tasks.items()}
        other.positions = dict(self.positions)
        other.velocities = dict(self.velocities)
        other.alive_set = set(self.alive_set)
        other.bodies = [dict(b) for b in self.bodies]
        other.meetings_called = dict(self.meetings_called)
        other.tasks_completed_total = self.tasks_completed_total

        other.kill_cooldown = self.kill_cooldown
        other.vent_cooldown = self.vent_cooldown
        other.emergency_cooldown = self.emergency_cooldown
        other.lights_out = self.lights_out
        other.sabotage_cooldown = self.sabotage_cooldown
        other.fixing_lights_progress = dict(self.fixing_lights_progress)

        other.is_doing_task = dict(self.is_doing_task)
        other.active_station = dict(self.active_station)
        other.task_progress = dict(self.task_progress)
        other.last_room = dict(self.last_room)
        other.prev_visible = {p: set(s) for p, s in self.prev_visible.items()}
        other.round_sightings = {
            p: [dict(s) for s in lst] for p, lst in self.round_sightings.items()
        }

        other.tick_count = self.tick_count
        other.map_time = self.map_time
        other.round_num = self.round_num
        other.round_start_time = self.round_start_time
        other.round_time = self.round_time
        other.phase = self.phase
        other.winner = self.winner
        other.win_reason = self.win_reason

        other.ticks_log = list(self.ticks_log)
        other.events_log = [dict(e) for e in self.events_log]
        other.meetings_log = [dict(m) for m in self.meetings_log]
        other.pending_votes = {p: dict(v) for p, v in self.pending_votes.items()}
        other.current_meeting = (
            dict(self.current_meeting) if self.current_meeting is not None else None
        )
        other.speaking_order = list(self.speaking_order)
        other.speaker_idx = self.speaker_idx
        other.last_meeting_summary = (
            dict(self.last_meeting_summary)
            if self.last_meeting_summary is not None
            else None
        )
        return other

    def alive_players(self) -> list[str]:
        return [p for p in PLAYERS if p in self.alive_set]

    def alive_crewmates_count(self) -> int:
        return sum(1 for p in self.alive_set if self.roles[p] == "CREWMATE")

    def _nearest_body(self, x: float, y: float, max_dist: float) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_d2 = max_dist * max_dist
        for b in self.bodies:
            d2 = (x - b["x"]) ** 2 + (y - b["y"]) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best = b
        return best

    def _nearest_kill_target(self, imp_id: str) -> str | None:
        ix, iy = self.positions[imp_id]
        best: str | None = None
        best_d2 = KILL_RANGE * KILL_RANGE
        for p in PLAYERS:
            if p != imp_id and p in self.alive_set and self.roles[p] == "CREWMATE":
                px, py = self.positions[p]
                d2 = (ix - px) ** 2 + (iy - py) ** 2
                if d2 <= best_d2:
                    best_d2 = d2
                    best = p
        return best

    def _nearest_vent(self, x: float, y: float) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_d2 = VENT_RANGE * VENT_RANGE
        for v in VENTS:
            d2 = (x - v["x"]) ** 2 + (y - v["y"]) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best = v
        return best

    def _nearest_actionable_station(self, player_id: str) -> dict[str, Any] | None:
        px, py = self.positions[player_id]
        best: dict[str, Any] | None = None
        best_d2 = INTERACT_RANGE * INTERACT_RANGE
        if self.roles[player_id] == "CREWMATE":
            for sid in self.incomplete_tasks[player_id]:
                st = STATION_BY_ID[sid]
                d2 = (px - st["x"]) ** 2 + (py - st["y"]) ** 2
                if d2 <= best_d2:
                    best_d2 = d2
                    best = st
        else:
            for st in TASK_STATIONS:
                d2 = (px - st["x"]) ** 2 + (py - st["y"]) ** 2
                if d2 <= best_d2:
                    best_d2 = d2
                    best = st
        return best

    def _append_sighting(
        self,
        player_id: str,
        text: str,
        kind: str,
        subject: str | None = None,
        room: str | None = None,
    ) -> None:
        entry = {
            "t": round(self.map_time, 1),
            "time_str": format_time(self.map_time),
            "kind": kind,
            "subject": subject,
            "room": room,
            "text": f"{format_time(self.map_time)} - {text}",
        }
        self.round_sightings[player_id].append(entry)

    def _effective_vision(self, player_id: str) -> float:
        if self.roles[player_id] == "IMPOSTOR":
            return IMPOSTOR_VISION
        return CREWMATE_LIGHTS_OUT_VISION if self.lights_out else CREWMATE_VISION

    def _finish_game(self, winner: str, win_reason: str) -> None:
        self.winner = winner
        self.win_reason = win_reason
        self.phase = "GAME_OVER"

    def _record_tick(
        self,
        alive_for_tick: list[str],
        clean_inputs: dict[str, dict[str, bool]],
        tick_actions: dict[str, str],
        vis_players_map: dict[str, list[str]],
        vis_bodies_map: dict[str, list[str]],
    ) -> None:
        if not self.record_ticks:
            return
        players_snap: dict[str, dict[str, Any]] = {}
        for p in alive_for_tick:
            px, py = self.positions[p]
            vx, vy = self.velocities[p]
            inp = clean_inputs[p]
            players_snap[p] = {
                "x": round(px, 1),
                "y": round(py, 1),
                "vx": round(vx, 1),
                "vy": round(vy, 1),
                "zone": get_zone(px, py),
                "keys": {
                    "w": 1 if inp["w"] else 0,
                    "a": 1 if inp["a"] else 0,
                    "s": 1 if inp["s"] else 0,
                    "d": 1 if inp["d"] else 0,
                },
                "action": tick_actions[p],
                "visible_players": vis_players_map[p],
                "visible_bodies": vis_bodies_map[p],
                "kill_cooldown": (
                    round(self.kill_cooldown, 2) if self.roles[p] == "IMPOSTOR" else 0.0
                ),
                "is_doing_task": self.is_doing_task[p],
                "near_task": self._nearest_actionable_station(p) is not None,
            }
        self.ticks_log.append(
            {
                "t": round(self.map_time, 1),
                "round": self.round_num,
                "lights_out": self.lights_out,
                "tasks_completed_total": self.tasks_completed_total,
                "players": players_snap,
            }
        )

    def _compute_vision_and_sightings(
        self,
        players_list: list[str],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Compute per-player LOS-occluded visible players/bodies and update sightings."""
        vis_players_map: dict[str, list[str]] = {}
        vis_bodies_map: dict[str, list[str]] = {}

        for p in players_list:
            px, py = self.positions[p]
            radius = self._effective_vision(p)
            r2 = radius * radius

            seen_p: list[str] = []
            for other in players_list:
                if other == p:
                    continue
                ox, oy = self.positions[other]
                if (px - ox) ** 2 + (py - oy) ** 2 <= r2 and has_line_of_sight(px, py, ox, oy):
                    seen_p.append(other)

            seen_b: list[str] = []
            for b in self.bodies:
                if (px - b["x"]) ** 2 + (py - b["y"]) ** 2 <= r2 and has_line_of_sight(
                    px, py, b["x"], b["y"]
                ):
                    seen_b.append(b["victim"])

            vis_players_map[p] = seen_p
            vis_bodies_map[p] = seen_b

            # Track newly seen players for Meeting Sightings sidebar
            seen_set = set(seen_p)
            newly_seen = seen_set - self.prev_visible[p]
            for other in sorted(newly_seen):
                ox, oy = self.positions[other]
                ozone = get_zone(ox, oy)
                self._append_sighting(
                    p,
                    f"Saw {other} in {ozone}",
                    kind="SAW_PLAYER",
                    subject=other,
                    room=ozone if ozone in ROOMS else None,
                )
            self.prev_visible[p] = seen_set

        return vis_players_map, vis_bodies_map

    def step(
        self,
        player_inputs: dict[str, dict[str, Any]] | None = None,
        dt: float = DEFAULT_DT,
    ) -> None:
        """Advance the 2D real-time simulation by `dt` seconds (no-op unless MAP)."""
        if self.phase != "MAP":
            return

        raw_inputs = player_inputs or {}
        alive_start = self.alive_players()
        clean_inputs: dict[str, dict[str, bool]] = {}
        for p in alive_start:
            src = raw_inputs.get(p) or {}
            clean_inputs[p] = {
                "w": bool(src.get("w", False)),
                "a": bool(src.get("a", False)),
                "s": bool(src.get("s", False)),
                "d": bool(src.get("d", False)),
                "interact": bool(src.get("interact", False)),
                "kill": bool(src.get("kill", False)),
                "vent": bool(src.get("vent", False)),
                "sabotage": bool(src.get("sabotage", False)),
            }

        self.tick_count += 1
        self.map_time = round(self.map_time + dt, 4)
        self.round_time = round(self.round_time + dt, 4)

        self.kill_cooldown = round(max(0.0, self.kill_cooldown - dt), 4)
        self.vent_cooldown = round(max(0.0, self.vent_cooldown - dt), 4)
        self.emergency_cooldown = round(max(0.0, self.emergency_cooldown - dt), 4)
        if not self.lights_out:
            self.sabotage_cooldown = round(max(0.0, self.sabotage_cooldown - dt), 4)

        tick_actions: dict[str, str] = {p: "NONE" for p in alive_start}

        # 1. Body report check (pauses map immediately)
        for p in alive_start:
            if clean_inputs[p]["interact"]:
                px, py = self.positions[p]
                body = self._nearest_body(px, py, INTERACT_RANGE)
                if body is not None:
                    tick_actions[p] = "REPORT"
                    for q in alive_start:
                        self.velocities[q] = (0.0, 0.0)
                        self.is_doing_task[q] = False
                        self.task_progress[q] = 0.0
                        self.active_station[q] = None
                    self.events_log.append(
                        {
                            "t": round(self.map_time, 1),
                            "round": self.round_num,
                            "type": "BODY_REPORT",
                            "player": p,
                            "victim": body["victim"],
                            "zone": body["zone"],
                        }
                    )
                    vis_p, vis_b = self._compute_vision_and_sightings(alive_start)
                    self._record_tick(alive_start, clean_inputs, tick_actions, vis_p, vis_b)
                    self._start_meeting(
                        trigger="BODY_REPORT",
                        caller=p,
                        victim=body["victim"],
                        body_pos=(body["x"], body["y"]),
                        body_zone=body["zone"],
                    )
                    return

        # 2. Check pending emergency button press (resolved after kills/movement)
        emergency_caller: str | None = None
        for p in alive_start:
            if (
                clean_inputs[p]["interact"]
                and self.emergency_cooldown <= 0.0
                and self.meetings_called[p] < 1
            ):
                px, py = self.positions[p]
                bx, by = EMERGENCY_BUTTON_POS
                if (px - bx) ** 2 + (py - by) ** 2 <= INTERACT_RANGE * INTERACT_RANGE:
                    emergency_caller = p
                    break

        # 3. Kill check (Impostor only)
        imp = self.impostor_id
        killed_target: str | None = None
        if imp in self.alive_set and clean_inputs[imp]["kill"] and self.kill_cooldown <= 0.0:
            target = self._nearest_kill_target(imp)
            if target is not None:
                killed_target = target
                tx, ty = self.positions[target]
                ix, iy = self.positions[imp]
                tzone = get_zone(tx, ty)

                self.alive_set.discard(target)
                self.velocities[target] = (0.0, 0.0)
                self.is_doing_task[target] = False
                self.task_progress[target] = 0.0
                self.active_station[target] = None

                self.bodies.append(
                    {
                        "victim": target,
                        "x": round(tx, 1),
                        "y": round(ty, 1),
                        "zone": tzone,
                        "t": round(self.map_time, 1),
                    }
                )
                self.kill_cooldown = POST_KILL_COOLDOWN
                self.is_doing_task[imp] = False
                self.task_progress[imp] = 0.0
                self.active_station[imp] = None
                tick_actions[imp] = "KILL"
                if emergency_caller == target:
                    emergency_caller = None

                witnesses: list[str] = []
                for w in alive_start:
                    if w in (imp, target) or w not in self.alive_set:
                        continue
                    wx, wy = self.positions[w]
                    wr2 = self._effective_vision(w) ** 2
                    saw_vic = (wx - tx) ** 2 + (wy - ty) ** 2 <= wr2 and has_line_of_sight(
                        wx, wy, tx, ty
                    )
                    saw_imp = (wx - ix) ** 2 + (wy - iy) ** 2 <= wr2 and has_line_of_sight(
                        wx, wy, ix, iy
                    )
                    if saw_vic or saw_imp:
                        witnesses.append(w)
                        self.events_log.append(
                            {
                                "t": round(self.map_time, 1),
                                "round": self.round_num,
                                "type": "SAW_KILL",
                                "player": w,
                                "suspect": imp,
                                "victim": target,
                                "zone": tzone,
                            }
                        )
                        self._append_sighting(
                            w,
                            f"SAW {imp} KILL {target} in {tzone}!",
                            kind="SAW_KILL",
                            subject=imp,
                            room=tzone if tzone in ROOMS else None,
                        )

                self.events_log.append(
                    {
                        "t": round(self.map_time, 1),
                        "round": self.round_num,
                        "type": "KILL",
                        "player": imp,
                        "victim": target,
                        "x": round(tx, 1),
                        "y": round(ty, 1),
                        "zone": tzone,
                        "witnesses": witnesses,
                    }
                )

        # 3b. Sabotage Lights check (Impostor only)
        if (
            imp in self.alive_set
            and clean_inputs[imp]["sabotage"]
            and not self.lights_out
            and self.sabotage_cooldown <= 0.0
        ):
            self.lights_out = True
            self.sabotage_cooldown = SABOTAGE_COOLDOWN
            tick_actions[imp] = "SABOTAGE"
            self.events_log.append(
                {
                    "t": round(self.map_time, 1),
                    "round": self.round_num,
                    "type": "SABOTAGE_LIGHTS",
                    "player": imp,
                }
            )

        # 4. Movement, Venting & Task Channeling for remaining alive players
        alive_now = self.alive_players()
        for p in alive_now:
            inp = clean_inputs[p]

            # Vent (Impostor only)
            if p == imp and inp["vent"] and self.vent_cooldown <= 0.0:
                px, py = self.positions[p]
                orig_vent = self._nearest_vent(px, py)
                if orig_vent is not None:
                    dest_vent = VENT_BY_ID[orig_vent["target_id"]]
                    self.positions[p] = (dest_vent["x"], dest_vent["y"])
                    self.velocities[p] = (0.0, 0.0)
                    self.is_doing_task[p] = False
                    self.task_progress[p] = 0.0
                    self.active_station[p] = None
                    self.vent_cooldown = VENT_COOLDOWN
                    self.last_room[p] = dest_vent["room"]
                    tick_actions[p] = "VENT"

                    vent_witnesses: list[str] = []
                    for w in alive_now:
                        if w == p:
                            continue
                        wx, wy = self.positions[w]
                        wr2 = self._effective_vision(w) ** 2
                        saw_orig = (
                            (wx - orig_vent["x"]) ** 2 + (wy - orig_vent["y"]) ** 2 <= wr2
                            and has_line_of_sight(wx, wy, orig_vent["x"], orig_vent["y"])
                        )
                        saw_dest = (
                            (wx - dest_vent["x"]) ** 2 + (wy - dest_vent["y"]) ** 2 <= wr2
                            and has_line_of_sight(wx, wy, dest_vent["x"], dest_vent["y"])
                        )
                        if saw_orig or saw_dest:
                            vent_witnesses.append(w)
                            seen_room = orig_vent["room"] if saw_orig else dest_vent["room"]
                            self.events_log.append(
                                {
                                    "t": round(self.map_time, 1),
                                    "round": self.round_num,
                                    "type": "SAW_VENT",
                                    "player": w,
                                    "suspect": p,
                                    "room": seen_room,
                                }
                            )
                            self._append_sighting(
                                w,
                                f"SAW {p} VENT in {seen_room}!",
                                kind="SAW_VENT",
                                subject=p,
                                room=seen_room,
                            )

                    self.events_log.append(
                        {
                            "t": round(self.map_time, 1),
                            "round": self.round_num,
                            "type": "VENT",
                            "player": p,
                            "from_room": orig_vent["room"],
                            "to_room": dest_vent["room"],
                            "witnesses": vent_witnesses,
                        }
                    )
                    continue

            # Continuous 8-directional movement
            mx = (1.0 if inp["d"] else 0.0) - (1.0 if inp["a"] else 0.0)
            my = (1.0 if inp["s"] else 0.0) - (1.0 if inp["w"] else 0.0)
            if mx != 0.0 and my != 0.0:
                inv_sq2 = 0.7071067811865476
                mx *= inv_sq2
                my *= inv_sq2

            vx = mx * PLAYER_SPEED
            vy = my * PLAYER_SPEED
            px, py = self.positions[p]
            nx, ny = move_with_sliding(px, py, vx * dt, vy * dt)
            self.positions[p] = (nx, ny)
            self.velocities[p] = (
                (nx - px) / dt if dt > 0 else 0.0,
                (ny - py) / dt if dt > 0 else 0.0,
            )

            zone_now = get_zone(nx, ny)
            if zone_now in ROOM_AABBS and zone_now != self.last_room[p]:
                self.last_room[p] = zone_now
                self.events_log.append(
                    {
                        "t": round(self.map_time, 1),
                        "round": self.round_num,
                        "type": "ROOM_ENTER",
                        "player": p,
                        "room": zone_now,
                    }
                )
                self._append_sighting(
                    p,
                    f"Entered {zone_now}",
                    kind="ROOM_ENTER",
                    room=zone_now,
                )

            # Moving cancels active task channeling and light-fixing
            if mx != 0.0 or my != 0.0:
                self.fixing_lights_progress[p] = 0.0
                if self.is_doing_task[p]:
                    self.is_doing_task[p] = False
                    self.task_progress[p] = 0.0
                    self.active_station[p] = None
            else:
                # Stationary: check Lights Out breaker fix first (for bots / hold-to-fix)
                lx, ly = LIGHTS_PANEL_POS
                near_lights = (
                    self.lights_out
                    and (nx - lx) ** 2 + (ny - ly) ** 2 <= INTERACT_RANGE * INTERACT_RANGE
                )
                if near_lights and (not self.is_human[p]) and (
                    inp["interact"] or self.fixing_lights_progress[p] > 0.0
                ):
                    self.fixing_lights_progress[p] = round(
                        self.fixing_lights_progress[p] + dt, 4
                    )
                    if self.fixing_lights_progress[p] >= BOT_FIX_LIGHTS_DURATION:
                        self.fix_lights(p)
                        tick_actions[p] = "FIX_LIGHTS"
                    continue

                # Stationary: initiate or advance task
                if inp["interact"] and not self.is_doing_task[p]:
                    st = self._nearest_actionable_station(p)
                    if st is not None:
                        self.is_doing_task[p] = True
                        self.active_station[p] = st["station_id"]
                        self.task_progress[p] = 0.0

                if self.is_doing_task[p] and self.active_station[p] is not None:
                    st = STATION_BY_ID[self.active_station[p]]
                    if (nx - st["x"]) ** 2 + (ny - st["y"]) ** 2 <= INTERACT_RANGE * INTERACT_RANGE:
                        if self.is_human[p]:
                            # Human players complete tasks via interactive mini-game modal
                            self.task_progress[p] = round(
                                min(TASK_DURATION * 0.92, self.task_progress[p] + dt), 4
                            )
                        else:
                            self.task_progress[p] = round(self.task_progress[p] + dt, 4)
                        if tick_actions[p] == "NONE":
                            tick_actions[p] = "DO_TASK"
                        if (not self.is_human[p]) and self.task_progress[p] >= TASK_DURATION:
                            sid = self.active_station[p]
                            self.is_doing_task[p] = False
                            self.task_progress[p] = 0.0
                            self.active_station[p] = None
                            if self.roles[p] == "CREWMATE" and sid in self.incomplete_tasks[p]:
                                self.incomplete_tasks[p].remove(sid)
                                self.tasks_completed_total += 1
                                self.events_log.append(
                                    {
                                        "t": round(self.map_time, 1),
                                        "round": self.round_num,
                                        "type": "TASK_COMPLETE",
                                        "player": p,
                                        "station_id": sid,
                                        "room": st["room"],
                                    }
                                )
                                self._append_sighting(
                                    p,
                                    f"Completed {st['label']} in {st['room']}",
                                    kind="TASK_COMPLETE",
                                    room=st["room"],
                                )
                    else:
                        self.is_doing_task[p] = False
                        self.task_progress[p] = 0.0
                        self.active_station[p] = None

        if emergency_caller is not None and tick_actions.get(emergency_caller) == "NONE":
            tick_actions[emergency_caller] = "MEETING"

        # 5. Vision, sightings, and 10 Hz trajectory tick capture
        vis_p, vis_b = self._compute_vision_and_sightings(alive_now)
        self._record_tick(alive_now, clean_inputs, tick_actions, vis_p, vis_b)

        # 6. Win conditions & Emergency Meeting transition
        if self.tasks_completed_total >= TASKS_WIN_THRESHOLD:
            self._finish_game("CREWMATES", "TASKS_COMPLETED")
            return
        if self.alive_crewmates_count() <= 1:
            self._finish_game("IMPOSTOR", "CREWMATES_ELIMINATED")
            return

        if emergency_caller is not None:
            self.meetings_called[emergency_caller] += 1
            for q in alive_now:
                self.velocities[q] = (0.0, 0.0)
                self.is_doing_task[q] = False
                self.task_progress[q] = 0.0
                self.active_station[q] = None
            self.events_log.append(
                {
                    "t": round(self.map_time, 1),
                    "round": self.round_num,
                    "type": "EMERGENCY_CALL",
                    "player": emergency_caller,
                    "zone": "Cafeteria",
                }
            )
            self._start_meeting(
                trigger="EMERGENCY",
                caller=emergency_caller,
                victim=None,
                body_pos=None,
                body_zone=None,
            )
            return

        if self.map_time >= MAX_MAP_TIME:
            self._finish_game("IMPOSTOR", "TIME_LIMIT")

    def _start_meeting(
        self,
        trigger: str,
        caller: str,
        victim: str | None,
        body_pos: tuple[float, float] | None,
        body_zone: str | None,
    ) -> None:
        alive_at_meeting = self.alive_players()
        caller_idx = alive_at_meeting.index(caller)
        self.speaking_order = (
            alive_at_meeting[caller_idx:] + alive_at_meeting[:caller_idx]
        )
        self.speaker_idx = 0
        self.pending_votes = {}
        self.current_meeting = {
            "meeting_id": len(self.meetings_log) + 1,
            "round": self.round_num,
            "t_called": round(self.map_time, 1),
            "round_start_t": round(self.round_start_time, 1),
            "trigger": trigger,
            "caller": caller,
            "victim": victim,
            "body_x": round(body_pos[0], 1) if body_pos is not None else None,
            "body_y": round(body_pos[1], 1) if body_pos is not None else None,
            "body_zone": body_zone,
            "alive_at_meeting": alive_at_meeting,
            "time_windows": get_round_time_windows(self.round_start_time, self.map_time),
            "quick_chat": [],
        }
        self.phase = "MEETING_CHAT"

    def current_speaker(self) -> str | None:
        if self.phase != "MEETING_CHAT":
            return None
        if self.speaker_idx < len(self.speaking_order):
            return self.speaking_order[self.speaker_idx]
        return None

    def submit_chat(
        self,
        player_id: str,
        chat_type: str,
        args: dict[str, Any] | None = None,
        rt_ms: float | int = 0,
    ) -> None:
        if self.phase != "MEETING_CHAT" or self.current_meeting is None:
            raise ValueError(f"Cannot submit chat in phase {self.phase}")
        expected = self.current_speaker()
        if player_id != expected:
            raise ValueError(f"Expected speaker {expected}, got {player_id}")
        if chat_type not in CHAT_TYPES:
            raise ValueError(f"Invalid chat_type: {chat_type}")

        raw_args = args or {}
        alive_at_meeting = self.current_meeting["alive_at_meeting"]
        valid_windows = self.current_meeting["time_windows"]
        clean_args: dict[str, Any]

        if chat_type == "CLAIM_LOC":
            room = str(raw_args.get("room", ""))
            twin = str(raw_args.get("time_window", ""))
            if room not in ROOMS:
                raise ValueError(f"Invalid room in CLAIM_LOC: {room}")
            if twin not in valid_windows and parse_time_window(twin) is None:
                raise ValueError(f"Invalid time_window in CLAIM_LOC: {twin}")
            clean_args = {"room": room, "time_window": twin}
        elif chat_type == "VOUCH":
            target = str(raw_args.get("target", ""))
            room = str(raw_args.get("room", ""))
            if target not in PLAYERS or target == player_id:
                raise ValueError(f"Invalid target in VOUCH: {target}")
            if room not in ROOMS:
                raise ValueError(f"Invalid room in VOUCH: {room}")
            clean_args = {"target": target, "room": room}
        elif chat_type == "ACCUSE":
            target = str(raw_args.get("target", ""))
            reason = str(raw_args.get("reason", ""))
            if target not in alive_at_meeting or target == player_id:
                raise ValueError(f"Invalid target in ACCUSE: {target}")
            if reason not in ACCUSE_REASONS:
                raise ValueError(f"Invalid reason in ACCUSE: {reason}")
            clean_args = {"target": target, "reason": reason}
        else:
            clean_args = {}

        self.current_meeting["quick_chat"].append(
            {
                "player": player_id,
                "order": self.speaker_idx,
                "type": chat_type,
                "args": clean_args,
                "rt_ms": int(round(float(rt_ms))) if isinstance(rt_ms, (int, float)) else 0,
            }
        )
        self.speaker_idx += 1
        if self.speaker_idx >= len(self.speaking_order):
            self.phase = "MEETING_VOTE"

    def submit_vote(
        self,
        player_id: str,
        target: str,
        rt_ms: float | int = 0,
    ) -> bool:
        if self.phase != "MEETING_VOTE" or self.current_meeting is None:
            raise ValueError(f"Cannot submit vote in phase {self.phase}")
        alive_at_meeting = self.current_meeting["alive_at_meeting"]
        if player_id not in alive_at_meeting:
            raise ValueError(f"Player {player_id} cannot vote in this meeting")
        if player_id in self.pending_votes:
            raise ValueError(f"Player {player_id} already voted")
        if target != "SKIP" and target not in alive_at_meeting:
            raise ValueError(f"Invalid vote target: {target}")

        self.pending_votes[player_id] = {
            "target": target,
            "rt_ms": int(round(float(rt_ms))) if isinstance(rt_ms, (int, float)) else 0,
        }

        if len(self.pending_votes) == len(alive_at_meeting):
            self._resolve_meeting_votes()
            return True
        return False

    def step_votes(
        self,
        votes: dict[str, str],
        rt_ms: dict[str, float | int] | None = None,
    ) -> None:
        if self.phase != "MEETING_VOTE" or self.current_meeting is None:
            raise ValueError(f"step_votes requires MEETING_VOTE phase, got {self.phase}")
        alive_at_meeting = self.current_meeting["alive_at_meeting"]
        missing = [p for p in alive_at_meeting if p not in votes]
        if missing:
            raise ValueError(f"Missing votes for: {missing}")
        rt_map = rt_ms or {}
        for p in alive_at_meeting:
            if p not in self.pending_votes:
                self.submit_vote(p, votes[p], rt_ms=rt_map.get(p, 0))

    def _resolve_meeting_votes(self) -> None:
        assert self.current_meeting is not None
        cm = self.current_meeting
        alive_at_meeting: list[str] = cm["alive_at_meeting"]

        counts: dict[str, int] = {}
        ordered_votes: dict[str, dict[str, Any]] = {}
        for p in alive_at_meeting:
            v = self.pending_votes[p]
            ordered_votes[p] = {"target": v["target"], "rt_ms": v["rt_ms"]}
            t = v["target"]
            counts[t] = counts.get(t, 0) + 1

        max_votes = max(counts.values())
        leaders = [t for t, c in counts.items() if c == max_votes]

        if len(leaders) == 1 and leaders[0] != "SKIP":
            ejected: str | None = leaders[0]
            ejected_role: str | None = self.roles[ejected]
            self.alive_set.discard(ejected)
        else:
            ejected = None
            ejected_role = None

        self.meetings_log.append(
            {
                "meeting_id": cm["meeting_id"],
                "round": cm["round"],
                "t_called": cm["t_called"],
                "round_start_t": cm["round_start_t"],
                "trigger": cm["trigger"],
                "caller": cm["caller"],
                "victim": cm["victim"],
                "body_x": cm["body_x"],
                "body_y": cm["body_y"],
                "body_zone": cm["body_zone"],
                "alive_at_meeting": list(alive_at_meeting),
                "quick_chat": list(cm["quick_chat"]),
                "votes": ordered_votes,
                "ejected": ejected,
                "ejected_role": ejected_role,
            }
        )
        self.last_meeting_summary = {
            "meeting_id": cm["meeting_id"],
            "t_called": cm["t_called"],
            "trigger": cm["trigger"],
            "caller": cm["caller"],
            "ejected": ejected,
            "ejected_role": ejected_role,
            "vote_counts": counts,
        }
        self.current_meeting = None
        self.pending_votes = {}
        self.speaking_order = []
        self.speaker_idx = 0

        if ejected is not None and ejected_role == "IMPOSTOR":
            self._finish_game("CREWMATES", "IMPOSTOR_EJECTED")
            return
        if self.tasks_completed_total >= TASKS_WIN_THRESHOLD:
            self._finish_game("CREWMATES", "TASKS_COMPLETED")
            return
        if self.alive_crewmates_count() <= 1:
            self._finish_game("IMPOSTOR", "CREWMATES_ELIMINATED")
            return
        if self.map_time >= MAX_MAP_TIME:
            self._finish_game("IMPOSTOR", "TIME_LIMIT")
            return

        # Reset map state for next round
        self.bodies.clear()
        for p in self.alive_set:
            self.positions[p] = SPAWN_POINTS[p]
            self.velocities[p] = (0.0, 0.0)
            self.is_doing_task[p] = False
            self.task_progress[p] = 0.0
            self.active_station[p] = None
            self.last_room[p] = "Cafeteria"
            self.prev_visible[p] = set()
            self.round_sightings[p] = []

        self.round_num += 1
        self.round_start_time = self.map_time
        self.round_time = 0.0
        self.kill_cooldown = POST_MEETING_KILL_COOLDOWN
        self.emergency_cooldown = EMERGENCY_COOLDOWN_START
        self.vent_cooldown = 0.0
        self.lights_out = False
        self.sabotage_cooldown = max(self.sabotage_cooldown, 10.0)
        self.fixing_lights_progress = {p: 0.0 for p in PLAYERS}
        self.phase = "MAP"

    def complete_task_station(self, player_id: str, station_id: str) -> bool:
        """Complete an interactive frontend task mini-game after validating proximity."""
        if self.phase != "MAP" or player_id not in self.alive_set:
            return False
        if station_id not in STATION_BY_ID:
            return False
        st = STATION_BY_ID[station_id]
        px, py = self.positions[player_id]
        if (px - st["x"]) ** 2 + (py - st["y"]) ** 2 > (INTERACT_RANGE + 15.0) ** 2:
            return False

        self.is_doing_task[player_id] = False
        self.task_progress[player_id] = 0.0
        self.active_station[player_id] = None

        if self.roles[player_id] == "CREWMATE" and station_id in self.incomplete_tasks[player_id]:
            self.incomplete_tasks[player_id].remove(station_id)
            self.tasks_completed_total += 1
            self.events_log.append(
                {
                    "t": round(self.map_time, 1),
                    "round": self.round_num,
                    "type": "TASK_COMPLETE",
                    "player": player_id,
                    "station_id": station_id,
                    "room": st["room"],
                }
            )
            self._append_sighting(
                player_id,
                f"Completed {st['label']} in {st['room']}",
                kind="TASK_COMPLETE",
                room=st["room"],
            )
            if self.tasks_completed_total >= TASKS_WIN_THRESHOLD:
                self._finish_game("CREWMATES", "TASKS_COMPLETED")
            return True
        return False

    def cancel_task(self, player_id: str) -> None:
        """Cancel active task channeling (e.g. player closed task popup)."""
        if player_id in self.is_doing_task:
            self.is_doing_task[player_id] = False
            self.task_progress[player_id] = 0.0
            self.active_station[player_id] = None

    def fix_lights(self, player_id: str) -> bool:
        """Restore lights from the Electrical breaker panel after proximity check."""
        if self.phase != "MAP" or not self.lights_out or player_id not in self.alive_set:
            return False
        px, py = self.positions[player_id]
        lx, ly = LIGHTS_PANEL_POS
        if (px - lx) ** 2 + (py - ly) ** 2 > (INTERACT_RANGE + 15.0) ** 2:
            return False

        self.lights_out = False
        self.fixing_lights_progress = {p: 0.0 for p in PLAYERS}
        self.events_log.append(
            {
                "t": round(self.map_time, 1),
                "round": self.round_num,
                "type": "FIX_LIGHTS",
                "player": player_id,
                "room": "Electrical",
            }
        )
        self._append_sighting(
            player_id,
            "Restored lights in Electrical",
            kind="FIX_LIGHTS",
            room="Electrical",
        )
        return True

    def get_observation(self, player_id: str) -> dict[str, Any]:
        """Strict per-player partial-observability snapshot.

        NEVER exposes off-vision players or bodies, hidden roles, or vents to Crewmates.
        """
        if player_id not in self.roles:
            raise ValueError(f"Unknown player_id: {player_id}")

        my_role = self.roles[player_id]
        is_imp = my_role == "IMPOSTOR"
        is_alive = player_id in self.alive_set
        px, py = self.positions[player_id]
        vx, vy = self.velocities[player_id]
        vision_radius = self._effective_vision(player_id)
        r2 = vision_radius * vision_radius

        map_geometry: dict[str, Any] = {
            "world_width": WORLD_WIDTH,
            "world_height": WORLD_HEIGHT,
            "player_radius": PLAYER_RADIUS,
            "rooms": {r: list(coords) for r, coords in ROOM_AABBS.items()},
            "corridors": {c: list(coords) for c, coords in CORRIDOR_VISUAL_RECTS.items()},
            "open_rects": [list(rect) for rect in OPEN_SPACE_RECTS],
            "emergency_button": list(EMERGENCY_BUTTON_POS),
            "lights_panel": list(LIGHTS_PANEL_POS),
            "task_stations": [dict(st) for st in TASK_STATIONS],
        }
        if is_imp:
            map_geometry["vents"] = [dict(v) for v in VENTS]

        visible_players: list[dict[str, Any]] = []
        visible_bodies: list[dict[str, Any]] = []
        if is_alive and self.phase != "GAME_OVER":
            for other in PLAYERS:
                if other == player_id or other not in self.alive_set:
                    continue
                ox, oy = self.positions[other]
                if (px - ox) ** 2 + (py - oy) ** 2 <= r2 and has_line_of_sight(px, py, ox, oy):
                    ovx, ovy = self.velocities[other]
                    visible_players.append(
                        {
                            "player_id": other,
                            "x": round(ox, 1),
                            "y": round(oy, 1),
                            "vx": round(ovx, 1),
                            "vy": round(ovy, 1),
                            "is_doing_task": self.is_doing_task[other],
                            "task_progress_ratio": (
                                round(min(1.0, self.task_progress[other] / TASK_DURATION), 3)
                                if self.is_doing_task[other]
                                else 0.0
                            ),
                        }
                    )
            for b in self.bodies:
                if (px - b["x"]) ** 2 + (py - b["y"]) ** 2 <= r2 and has_line_of_sight(
                    px, py, b["x"], b["y"]
                ):
                    visible_bodies.append(dict(b))

        near_body = (
            self._nearest_body(px, py, INTERACT_RANGE)
            if is_alive and self.phase == "MAP"
            else None
        )
        near_station = (
            self._nearest_actionable_station(player_id)
            if is_alive and self.phase == "MAP"
            else None
        )
        lx, ly = LIGHTS_PANEL_POS
        near_lights_panel = (
            is_alive
            and self.phase == "MAP"
            and self.lights_out
            and (px - lx) ** 2 + (py - ly) ** 2 <= INTERACT_RANGE * INTERACT_RANGE
        )
        bx, by = EMERGENCY_BUTTON_POS
        near_button = (
            is_alive
            and self.phase == "MAP"
            and (px - bx) ** 2 + (py - by) ** 2 <= INTERACT_RANGE * INTERACT_RANGE
        )
        can_call_meeting = (
            near_button
            and self.emergency_cooldown <= 0.0
            and self.meetings_called[player_id] < 1
        )

        meeting_view: dict[str, Any] | None = None
        if self.current_meeting is not None:
            cm = self.current_meeting
            meeting_view = {
                "meeting_id": cm["meeting_id"],
                "round": cm["round"],
                "t_called": cm["t_called"],
                "trigger": cm["trigger"],
                "caller": cm["caller"],
                "victim": cm["victim"],
                "body_zone": cm["body_zone"],
                "alive_at_meeting": list(cm["alive_at_meeting"]),
                "speaking_order": list(self.speaking_order),
                "current_speaker": self.current_speaker(),
                "time_windows": list(cm["time_windows"]),
                "quick_chat": [
                    {
                        "player": c["player"],
                        "order": c["order"],
                        "type": c["type"],
                        "args": dict(c["args"]),
                        "text": format_speech_act(c["type"], c["args"]),
                    }
                    for c in cm["quick_chat"]
                ],
                "votes_submitted_count": len(self.pending_votes),
                "votes_required_count": len(cm["alive_at_meeting"]),
            }

        obs: dict[str, Any] = {
            "game_id": self.game_id,
            "map_version": MAP_VERSION,
            "phase": self.phase,
            "round": self.round_num,
            "map_time": round(self.map_time, 1),
            "max_map_time": MAX_MAP_TIME,
            "round_time": round(self.round_time, 1),
            "my_player": player_id,
            "my_role": my_role,
            "is_alive": is_alive,
            "x": round(px, 1),
            "y": round(py, 1),
            "vx": round(vx, 1),
            "vy": round(vy, 1),
            "zone": get_zone(px, py),
            "vision_radius": vision_radius,
            "is_doing_task": self.is_doing_task[player_id],
            "task_duration": TASK_DURATION,
            "task_progress_ratio": (
                round(min(1.0, self.task_progress[player_id] / TASK_DURATION), 3)
                if self.is_doing_task[player_id]
                else 0.0
            ),
            "assigned_tasks": [dict(st) for st in self.assigned_tasks[player_id]],
            "incomplete_tasks": sorted(self.incomplete_tasks[player_id]),
            "tasks_completed_total": self.tasks_completed_total,
            "tasks_win_threshold": TASKS_WIN_THRESHOLD,
            "alive_players": self.alive_players(),
            "dead_players": [p for p in PLAYERS if p not in self.alive_set],
            "visible_players": visible_players,
            "visible_bodies": visible_bodies,
            "can_report": near_body is not None,
            "can_call_meeting": can_call_meeting,
            "near_button": near_button,
            "emergency_cooldown": round(self.emergency_cooldown, 1),
            "meetings_called_by_me": self.meetings_called[player_id],
            "can_start_task": near_station is not None and not self.is_doing_task[player_id],
            "near_station_id": near_station["station_id"] if near_station is not None else None,
            "near_station_label": near_station["label"] if near_station is not None else None,
            "near_station_room": near_station["room"] if near_station is not None else None,
            "active_station_id": self.active_station[player_id],
            "lights_out": self.lights_out,
            "near_lights_panel": near_lights_panel,
            "my_recent_sightings": list(self.round_sightings[player_id][-15:]),
            "map_geometry": map_geometry,
            "meeting": meeting_view,
            "has_submitted_vote": (
                self.phase == "MEETING_VOTE" and player_id in self.pending_votes
            ),
            "last_meeting_summary": (
                dict(self.last_meeting_summary)
                if self.last_meeting_summary is not None
                else None
            ),
            "game_over": (
                {
                    "winner": self.winner,
                    "win_reason": self.win_reason,
                    "impostor": self.impostor_id,
                    "total_time": round(self.map_time, 1),
                    "total_ticks": self.tick_count,
                }
                if self.phase == "GAME_OVER"
                else None
            ),
        }

        if is_imp:
            kill_target = (
                self._nearest_kill_target(player_id)
                if is_alive and self.phase == "MAP" and self.kill_cooldown <= 0.0
                else None
            )
            near_vent = (
                self._nearest_vent(px, py) if is_alive and self.phase == "MAP" else None
            )
            obs["kill_cooldown"] = round(self.kill_cooldown, 1)
            obs["vent_cooldown"] = round(self.vent_cooldown, 1)
            obs["sabotage_cooldown"] = round(self.sabotage_cooldown, 1)
            obs["can_sabotage"] = (
                is_alive
                and self.phase == "MAP"
                and not self.lights_out
                and self.sabotage_cooldown <= 0.0
            )
            obs["can_kill_target"] = kill_target
            obs["can_vent"] = near_vent is not None and self.vent_cooldown <= 0.0

        return obs

    def to_game_log(self) -> dict[str, Any]:
        """Export complete ground-truth JSON log (10 Hz trajectories, events, meetings)."""
        if self.phase != "GAME_OVER":
            raise ValueError("Game is not finished yet; cannot export ground-truth log")
        return {
            "game_id": self.game_id,
            "timestamp_utc": self.timestamp_utc,
            "map_version": MAP_VERSION,
            "tick_rate": TICK_RATE,
            "players": {
                p: {
                    "role": self.roles[p],
                    "is_human": self.is_human[p],
                    "assigned_tasks": [st["station_id"] for st in self.assigned_tasks[p]],
                }
                for p in PLAYERS
            },
            "winner": self.winner,
            "win_reason": self.win_reason,
            "total_time": round(self.map_time, 1),
            "total_ticks": self.tick_count,
            "ticks": self.ticks_log,
            "events": self.events_log,
            "meetings": self.meetings_log,
        }


# ---------------------------------------------------------------------------
# Smooth 2D Waypoint Bot AI (strictly observation-only)
# ---------------------------------------------------------------------------

def _steer_toward(
    x: float,
    y: float,
    tx: float,
    ty: float,
    tolerance: float = 10.0,
) -> tuple[bool, bool, bool, bool]:
    """Compute held `(w, a, s, d)` keys toward target `(tx, ty)` with corridor alignment."""
    zone = get_zone(x, y)
    aim_x, aim_y = tx, ty

    if zone.startswith("Corridor_"):
        x1, y1, x2, y2 = CORRIDOR_AABBS[zone]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        is_horizontal = (x2 - x1) > (y2 - y1)
        if is_horizontal and abs(y - cy) > 10.0:
            aim_y = cy
        elif (not is_horizontal) and abs(x - cx) > 10.0:
            aim_x = cx

    dx = aim_x - x
    dy = aim_y - y
    w = dy < -tolerance
    s = dy > tolerance
    a = dx < -tolerance
    d = dx > tolerance
    return (w, a, s, d)


def _resolve_dest_room(cur_zone: str, dest_zone: str, dest_x: float, dest_y: float) -> str:
    if dest_zone in ROOM_AABBS:
        return dest_zone
    if dest_zone in CORRIDOR_LINKS:
        r1, r2 = CORRIDOR_LINKS[dest_zone]
        if cur_zone == r1:
            return r2
        if cur_zone == r2:
            return r1
        c1, c2 = ROOM_CENTERS[r1], ROOM_CENTERS[r2]
        d1 = (dest_x - c1[0]) ** 2 + (dest_y - c1[1]) ** 2
        d2 = (dest_x - c2[0]) ** 2 + (dest_y - c2[1]) ** 2
        return r1 if d1 <= d2 else r2
    return "Cafeteria"


def _route_toward_room_point(
    x: float,
    y: float,
    dest_zone: str,
    dest_x: float,
    dest_y: float,
) -> tuple[bool, bool, bool, bool]:
    """Route smoothly along room/corridor portals from `(x, y)` to `(dest_x, dest_y)`."""
    zone = get_zone(x, y)
    if zone == dest_zone:
        return _steer_toward(x, y, dest_x, dest_y, tolerance=8.0)

    dest_room = _resolve_dest_room(zone, dest_zone, dest_x, dest_y)
    if zone == dest_room:
        return _steer_toward(x, y, dest_x, dest_y, tolerance=8.0)

    if zone in ROOM_AABBS:
        cur_room = zone
        next_room = _ROOM_NEXT_HOP[dest_room].get(cur_room)
        if next_room is None:
            return _steer_toward(x, y, dest_x, dest_y)
        px, py = DOORWAY_PORTALS[(cur_room, next_room)]
        nx, ny = DOORWAY_PORTALS[(next_room, cur_room)]
        # Aim across the doorway into the next room's portal
        mid_x = 0.35 * px + 0.65 * nx
        mid_y = 0.35 * py + 0.65 * ny
        # First align perpendicular to doorway inside current room if far from doorway line
        if abs(px - nx) > abs(py - ny):
            # Horizontal corridor: align Y before pushing X through doorway
            if abs(y - py) > 18.0:
                return _steer_toward(x, y, px, py, tolerance=8.0)
        else:
            # Vertical corridor: align X before pushing Y through doorway
            if abs(x - px) > 18.0:
                return _steer_toward(x, y, px, py, tolerance=8.0)
        return _steer_toward(x, y, mid_x, mid_y, tolerance=8.0)

    if zone in CORRIDOR_LINKS:
        r1, r2 = CORRIDOR_LINKS[zone]
        # Pick whichever connected room is closer in hops to dest_room
        def hops_to(start_r: str) -> int:
            c = start_r
            h = 0
            while c != dest_room and h < 10:
                nxt = _ROOM_NEXT_HOP[dest_room].get(c)
                if nxt is None:
                    return 99
                c = nxt
                h += 1
            return h

        exit_room = r1 if hops_to(r1) <= hops_to(r2) else r2
        rcx, rcy = ROOM_CENTERS[exit_room]
        return _steer_toward(x, y, rcx, rcy, tolerance=8.0)

    return _steer_toward(x, y, dest_x, dest_y)


def choose_bot_step_input(
    obs: dict[str, Any],
    bot_mem: dict[str, Any],
    rng: random.Random | None = None,
) -> dict[str, bool]:
    """Generate natural real-time 10 Hz key/trigger inputs for one bot using ONLY `obs`."""
    r = rng or random
    inp = _empty_input()
    if not obs["is_alive"] or obs["phase"] != "MAP":
        return inp

    # Reset per-round navigation memory
    if bot_mem.get("round") != obs["round"]:
        bot_mem["round"] = obs["round"]
        bot_mem["pause_ticks"] = r.randint(5, 20)
        bot_mem["entry_walk_ticks"] = 0
        bot_mem["pending_pause"] = 0
        bot_mem["last_named_room"] = "Cafeteria"
        bot_mem["prev_inc_count"] = len(obs["incomplete_tasks"])
        bot_mem["wander_room"] = (
            r.choice(TASK_ROOMS)
            if obs["my_role"] == "CREWMATE" and r.random() < 0.35
            else None
        )
        bot_mem["current_task_id"] = None
        bot_mem["patrol_room"] = None
        bot_mem["fake_elapsed"] = 0
        bot_mem["fake_target_ticks"] = 0
        bot_mem["fake_cooldown_ticks"] = 0
        bot_mem["flee_ticks"] = 0
        bot_mem["stalk_pos"] = None
        bot_mem["stalk_zone"] = None
        bot_mem["stalk_ticks"] = 0
        bot_mem["respond_lights"] = False
        bot_mem["lights_was_out"] = False

    px, py = obs["x"], obs["y"]

    # 1. Body reporting (Crewmates always; Impostor avoids fresh self-reports)
    if obs["can_report"]:
        if obs["my_role"] == "CREWMATE" or (
            bot_mem.get("flee_ticks", 0) <= 0 and r.random() < 0.12
        ):
            inp["interact"] = True
            return inp

    if obs["visible_bodies"] and obs["my_role"] == "CREWMATE":
        b = obs["visible_bodies"][0]
        w, a, s, d = _route_toward_room_point(px, py, get_zone(b["x"], b["y"]), b["x"], b["y"])
        inp["w"], inp["a"], inp["s"], inp["d"] = w, a, s, d
        return inp

    # 2. Crewmate Bot: realistic pauses, 35% room wandering, and 5.0s task channeling
    if obs["my_role"] == "CREWMATE":
        saw_crime = any(
            sg["kind"] in ("SAW_KILL", "SAW_VENT") for sg in obs["my_recent_sightings"]
        )
        if saw_crime and obs["meetings_called_by_me"] < 1:
            if obs["can_call_meeting"]:
                inp["interact"] = True
                return inp
            bx, by = EMERGENCY_BUTTON_POS
            w, a, s, d = _route_toward_room_point(px, py, "Cafeteria", bx, by)
            inp["w"], inp["a"], inp["s"], inp["d"] = w, a, s, d
            return inp

        if obs["is_doing_task"]:
            return inp

        # Respond to Sabotage Lights with 60% chance per outage
        if obs.get("lights_out"):
            if not bot_mem.get("lights_was_out"):
                bot_mem["lights_was_out"] = True
                bot_mem["respond_lights"] = r.random() < 0.65
            if bot_mem.get("respond_lights"):
                if obs.get("near_lights_panel"):
                    inp["interact"] = True
                    return inp
                lx, ly = LIGHTS_PANEL_POS
                w, a, s, d = _route_toward_room_point(px, py, "Electrical", lx, ly)
                inp["w"], inp["a"], inp["s"], inp["d"] = w, a, s, d
                return inp
        else:
            bot_mem["lights_was_out"] = False
            bot_mem["respond_lights"] = False

        # Detect task completion -> 1.5s-3.5s pause + 35% chance to wander
        cur_inc_count = len(obs["incomplete_tasks"])
        if cur_inc_count < bot_mem.get("prev_inc_count", cur_inc_count):
            bot_mem["prev_inc_count"] = cur_inc_count
            bot_mem["current_task_id"] = None
            bot_mem["pause_ticks"] = r.randint(15, 35)
            if r.random() < 0.35:
                candidates = [rm for rm in ROOMS if rm != obs["zone"]]
                bot_mem["wander_room"] = r.choice(candidates)

        # Detect entering a new named room -> walk inside briefly, then 1.5s-3.5s look-around pause
        if obs["zone"] in ROOM_AABBS and obs["zone"] != bot_mem.get("last_named_room"):
            bot_mem["last_named_room"] = obs["zone"]
            bot_mem["entry_walk_ticks"] = 5
            bot_mem["pending_pause"] = r.randint(15, 35)

        if bot_mem.get("entry_walk_ticks", 0) > 0:
            bot_mem["entry_walk_ticks"] -= 1
            if bot_mem["entry_walk_ticks"] == 0 and bot_mem.get("pending_pause", 0) > 0:
                bot_mem["pause_ticks"] = bot_mem["pending_pause"]
                bot_mem["pending_pause"] = 0

        if bot_mem.get("pause_ticks", 0) > 0:
            bot_mem["pause_ticks"] -= 1
            return inp

        # Optional 35% wander detour before heading to next task
        wander_room = bot_mem.get("wander_room")
        if wander_room:
            wcx, wcy = ROOM_CENTERS[wander_room]
            if obs["zone"] == wander_room and math.hypot(px - wcx, py - wcy) < 48.0:
                bot_mem["wander_room"] = None
            else:
                w, a, s, d = _route_toward_room_point(px, py, wander_room, wcx, wcy)
                inp["w"], inp["a"], inp["s"], inp["d"] = w, a, s, d
                return inp

        # Assigned tasks
        incomplete = set(obs["incomplete_tasks"])
        if incomplete:
            if obs["can_start_task"]:
                inp["interact"] = True
                return inp
            cur_tid = bot_mem.get("current_task_id")
            if cur_tid not in incomplete:
                cur_tid = r.choice(sorted(incomplete))
                bot_mem["current_task_id"] = cur_tid
            target_st = next(
                st for st in obs["assigned_tasks"] if st["station_id"] == cur_tid
            )
            w, a, s, d = _route_toward_room_point(
                px, py, target_st["room"], target_st["x"], target_st["y"]
            )
            inp["w"], inp["a"], inp["s"], inp["d"] = w, a, s, d
            return inp

        # Patrol rooms once all 4 tasks are complete
        patrol_room = bot_mem.get("patrol_room")
        if not patrol_room or (
            obs["zone"] == patrol_room
            and math.hypot(px - ROOM_CENTERS[patrol_room][0], py - ROOM_CENTERS[patrol_room][1]) < 45.0
        ):
            patrol_room = r.choice(TASK_ROOMS)
            bot_mem["patrol_room"] = patrol_room
        tx, ty = ROOM_CENTERS[patrol_room]
        w, a, s, d = _route_toward_room_point(px, py, patrol_room, tx, ty)
        inp["w"], inp["a"], inp["s"], inp["d"] = w, a, s, d
        return inp

    # 3. Impostor Bot: natural room pauses, 3-5s task faking, stalking (CD <= 2.0s), isolated kills
    if bot_mem.get("fake_cooldown_ticks", 0) > 0:
        bot_mem["fake_cooldown_ticks"] -= 1
    if bot_mem.get("flee_ticks", 0) > 0:
        bot_mem["flee_ticks"] -= 1
    if bot_mem.get("stalk_ticks", 0) > 0:
        bot_mem["stalk_ticks"] -= 1

    kcd = obs.get("kill_cooldown", 1.0)
    vis_players = obs["visible_players"]

    # Trigger Sabotage Lights when available and ready to hunt
    if obs.get("can_sabotage") and kcd <= 4.0 and r.random() < 0.08:
        inp["sabotage"] = True
        return inp

    # Execute kill if ready and isolated (or game-winning / late-round fallback)
    if obs.get("can_kill_target") is not None and kcd <= 0.0:
        isolated_kill = len(vis_players) <= 1 and r.random() < 0.45
        clutch_kill = len(obs["alive_players"]) <= 3
        late_fallback = obs["round_time"] > 55.0 and r.random() < 0.25
        if isolated_kill or clutch_kill or late_fallback:
            inp["kill"] = True
            bot_mem["flee_ticks"] = 50
            bot_mem["stalk_ticks"] = 0
            bot_mem["pause_ticks"] = 0
            escape_rooms = [rm for rm in TASK_ROOMS if rm != obs["zone"]]
            bot_mem["patrol_room"] = r.choice(escape_rooms)
            return inp

    # Post-kill getaway / vent escape
    if bot_mem.get("flee_ticks", 0) > 0:
        if obs.get("can_vent") and len(vis_players) == 0:
            inp["vent"] = True
            bot_mem["flee_ticks"] = 0
            return inp
        esc = bot_mem.get("patrol_room") or r.choice(TASK_ROOMS)
        bot_mem["patrol_room"] = esc
        tx, ty = ROOM_CENTERS[esc]
        w, a, s, d = _route_toward_room_point(px, py, esc, tx, ty)
        inp["w"], inp["a"], inp["s"], inp["d"] = w, a, s, d
        return inp

    # Fake standing at a task station for 3.0s-5.0s when watched
    if obs["is_doing_task"]:
        if kcd <= 0.0 and len(vis_players) == 1:
            bot_mem["fake_elapsed"] = 0
        else:
            bot_mem["fake_elapsed"] = bot_mem.get("fake_elapsed", 0) + 1
            if bot_mem["fake_elapsed"] < bot_mem.get("fake_target_ticks", 40):
                return inp
            bot_mem["fake_elapsed"] = 0
            bot_mem["fake_cooldown_ticks"] = 90
            esc = r.choice([rm for rm in ROOMS if rm != obs["zone"]])
            bot_mem["patrol_room"] = esc
            tx, ty = ROOM_CENTERS[esc]
            w, a, s, d = _route_toward_room_point(px, py, esc, tx, ty)
            inp["w"], inp["a"], inp["s"], inp["d"] = w, a, s, d
            return inp

    # Actively stalk an isolated Crewmate when kill_cooldown <= 2.0s
    if kcd <= 2.0:
        if len(vis_players) == 1:
            vp = vis_players[0]
            vzone = get_zone(vp["x"], vp["y"])
            bot_mem["stalk_pos"] = (vp["x"], vp["y"])
            bot_mem["stalk_zone"] = vzone
            bot_mem["stalk_ticks"] = 25
            bot_mem["pause_ticks"] = 0
            w, a, s, d = _route_toward_room_point(px, py, vzone, vp["x"], vp["y"])
            inp["w"], inp["a"], inp["s"], inp["d"] = w, a, s, d
            return inp
        if len(vis_players) == 0 and bot_mem.get("stalk_ticks", 0) > 0 and bot_mem.get("stalk_pos"):
            sx, sy = bot_mem["stalk_pos"]
            sz = bot_mem["stalk_zone"] or get_zone(sx, sy)
            w, a, s, d = _route_toward_room_point(px, py, sz, sx, sy)
            inp["w"], inp["a"], inp["s"], inp["d"] = w, a, s, d
            return inp

    # Natural room-entry pause while patrolling on cooldown
    if obs["zone"] in ROOM_AABBS and obs["zone"] != bot_mem.get("last_named_room"):
        bot_mem["last_named_room"] = obs["zone"]
        bot_mem["entry_walk_ticks"] = 5
        bot_mem["pending_pause"] = r.randint(10, 25)

    if bot_mem.get("entry_walk_ticks", 0) > 0:
        bot_mem["entry_walk_ticks"] -= 1
        if bot_mem["entry_walk_ticks"] == 0 and bot_mem.get("pending_pause", 0) > 0:
            bot_mem["pause_ticks"] = bot_mem["pending_pause"]
            bot_mem["pending_pause"] = 0

    # Initiate a 3.0s-5.0s fake task if another player is in vision range
    can_consider_fake = (
        len(vis_players) >= 1
        and bot_mem.get("fake_cooldown_ticks", 0) <= 0
        and (kcd > 2.0 or len(vis_players) >= 2)
    )
    if can_consider_fake:
        bot_mem["pause_ticks"] = 0
        if obs["can_start_task"]:
            bot_mem["fake_elapsed"] = 0
            bot_mem["fake_target_ticks"] = r.randint(30, 50)
            inp["interact"] = True
            return inp
        if obs["zone"] in TASK_ROOMS:
            nearest_st = min(
                STATIONS_BY_ROOM[obs["zone"]],
                key=lambda st: (px - st["x"]) ** 2 + (py - st["y"]) ** 2,
            )
            w, a, s, d = _route_toward_room_point(
                px, py, nearest_st["room"], nearest_st["x"], nearest_st["y"]
            )
            inp["w"], inp["a"], inp["s"], inp["d"] = w, a, s, d
            return inp

    if bot_mem.get("pause_ticks", 0) > 0:
        bot_mem["pause_ticks"] -= 1
        return inp

    # Patrol task rooms to catch isolated Crewmates
    patrol_room = bot_mem.get("patrol_room")
    if not patrol_room or (
        obs["zone"] == patrol_room
        and math.hypot(px - ROOM_CENTERS[patrol_room][0], py - ROOM_CENTERS[patrol_room][1]) < 45.0
    ):
        patrol_room = r.choice(TASK_ROOMS)
        bot_mem["patrol_room"] = patrol_room
    tx, ty = ROOM_CENTERS[patrol_room]
    w, a, s, d = _route_toward_room_point(px, py, patrol_room, tx, ty)
    inp["w"], inp["a"], inp["s"], inp["d"] = w, a, s, d
    return inp


def choose_bot_chat(
    obs: dict[str, Any],
    rng: random.Random | None = None,
) -> tuple[str, dict[str, Any]]:
    """Choose a valid Quick Chat statement using only `obs`."""
    r = rng or random
    m = obs["meeting"]
    assert m is not None
    me = obs["my_player"]
    alive = [p for p in m["alive_at_meeting"] if p != me]
    sightings = obs["my_recent_sightings"]
    windows = m["time_windows"]

    # Immediately accuse witnessed killer/venter
    for sg in reversed(sightings):
        if sg["kind"] in ("SAW_KILL", "SAW_VENT") and sg["subject"] in alive:
            return ("ACCUSE", {"target": sg["subject"], "reason": sg["kind"]})

    if obs["my_role"] == "CREWMATE":
        options: list[tuple[str, dict[str, Any]]] = []
        room_events = [sg for sg in sightings if sg.get("room") in ROOMS]
        if room_events and windows:
            ev = r.choice(room_events)
            # Match time window containing ev['t']
            matched_win = windows[-1]
            for w_str in windows:
                bounds = parse_time_window(w_str)
                if bounds and bounds[0] <= ev["t"] <= bounds[1]:
                    matched_win = w_str
                    break
            if ev["kind"] == "SAW_PLAYER" and ev.get("subject") in PLAYERS and ev["subject"] != me:
                options.append(("VOUCH", {"target": ev["subject"], "room": ev["room"]}))
            else:
                options.append(("CLAIM_LOC", {"room": ev["room"], "time_window": matched_win}))
        if alive:
            options.append(
                ("ACCUSE", {"target": r.choice(alive), "reason": r.choice(("BAD_ALIBI", "SUSPICIOUS"))})
            )
        options.append(("PASS", {}))
        return r.choice(options)

    # Impostor bot mixes deceptive alibis / accusations
    roll = r.random()
    if roll < 0.5 and windows:
        return ("CLAIM_LOC", {"room": r.choice(TASK_ROOMS), "time_window": r.choice(windows)})
    if roll < 0.75 and alive:
        return ("ACCUSE", {"target": r.choice(alive), "reason": r.choice(("BAD_ALIBI", "SUSPICIOUS"))})
    return ("PASS", {})


def choose_bot_vote(
    obs: dict[str, Any],
    rng: random.Random | None = None,
) -> str:
    """Choose a valid vote target using only `obs`."""
    r = rng or random
    m = obs["meeting"]
    assert m is not None
    me = obs["my_player"]
    alive = [p for p in m["alive_at_meeting"] if p != me]
    for sg in reversed(obs["my_recent_sightings"]):
        if sg["kind"] in ("SAW_KILL", "SAW_VENT") and sg["subject"] in alive:
            return sg["subject"]
    # React to strong witness accusations from Quick Chat, otherwise favor SKIP unless suspicious
    if obs["my_role"] == "CREWMATE":
        for c in m["quick_chat"]:
            if c["type"] == "ACCUSE":
                cargs = c.get("args", {})
                if cargs.get("reason") in ("SAW_KILL", "SAW_VENT") and cargs.get("target") in alive:
                    return cargs["target"]
        if r.random() < 0.65:
            return "SKIP"
    return r.choice(alive + ["SKIP"])


def simulate_game(
    seed: int | None = None,
    is_human: dict[str, bool] | None = None,
    record_ticks: bool = True,
) -> dict[str, Any]:
    """Run a complete 10 Hz headless 2D game from t=0 to GAME_OVER."""
    engine = MiniAmongUsEngine(seed=seed, is_human=is_human, record_ticks=record_ticks)
    bot_mems: dict[str, dict[str, Any]] = {p: {} for p in PLAYERS}
    while engine.phase != "GAME_OVER":
        if engine.phase == "MAP":
            inputs = {
                p: choose_bot_step_input(engine.get_observation(p), bot_mems[p], engine.rng)
                for p in engine.alive_players()
            }
            engine.step(inputs, dt=DEFAULT_DT)
        elif engine.phase == "MEETING_CHAT":
            speaker = engine.current_speaker()
            assert speaker is not None
            ctype, cargs = choose_bot_chat(engine.get_observation(speaker), engine.rng)
            engine.submit_chat(speaker, ctype, cargs)
        elif engine.phase == "MEETING_VOTE":
            assert engine.current_meeting is not None
            votes = {
                p: choose_bot_vote(engine.get_observation(p), engine.rng)
                for p in engine.current_meeting["alive_at_meeting"]
            }
            engine.step_votes(votes)
        else:
            raise RuntimeError(f"Unexpected phase: {engine.phase}")
    return engine.to_game_log()
