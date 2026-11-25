"""CLI verification & benchmark suite for real-time 10 Hz 2D Mini-Among Us.

Verifies:
1. Pure-stdlib AST import check on `engine.py`.
2. 2D AABB geometry, diagonal wall sliding, `get_zone(x, y)`, and real-time cooldowns
   (10s/12s kill, 2.5s task + move-cancel, 2s vent, 15s emergency, SAW_KILL/SAW_VENT).
3. Continuous Logical Agent flags (`logic_impossible_travel`, `logic_witness_contradiction`).
4. Strict partial-observability non-leakage on Engine + FastAPI `/api/state`.
5. 10 full headless bot games at max speed (zero out-of-bounds, temp JSONL + CSV export).
"""

from __future__ import annotations

import ast
import asyncio
import csv
import json
from pathlib import Path
import sys
import tempfile
import time

import httpx

from engine import (
    CREWMATE_LIGHTS_OUT_VISION,
    CREWMATE_VISION,
    DEFAULT_DT,
    EMERGENCY_BUTTON_POS,
    IMPOSTOR_VISION,
    INITIAL_KILL_COOLDOWN,
    KILL_RANGE,
    LIGHTS_PANEL_POS,
    MiniAmongUsEngine,
    PLAYERS,
    POST_KILL_COOLDOWN,
    POST_MEETING_KILL_COOLDOWN,
    ROOM_AABBS,
    ROOM_CENTERS,
    SABOTAGE_COOLDOWN,
    TASK_DURATION,
    TASK_WIN_THRESHOLD,
    TASKS_PER_CREWMATE,
    VENTS,
    get_zone,
    has_line_of_sight,
    is_walkable,
    move_with_sliding,
    simulate_game,
)
from export_features import (
    KEYSTROKE_TRAJECTORY_COLUMNS,
    MEETING_FEATURE_COLUMNS,
    export_all,
    extract_meeting_rows,
)
import server


def test_engine_stdlib_only() -> None:
    """Verify engine.py uses 100% Python stdlib modules only."""
    source = Path("engine.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    stdlib = sys.stdlib_module_names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top in stdlib, f"Non-stdlib import in engine.py: {top}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            assert top in stdlib, f"Non-stdlib import in engine.py: {top}"


def test_geometry_collision_and_zones() -> None:
    """Verify 2D AABBs, wall sliding, and zone lookups."""
    for r, (cx, cy) in ROOM_CENTERS.items():
        assert is_walkable(cx, cy)
        assert get_zone(cx, cy) == r

    # Corridor check
    assert is_walkable(395.0, 190.0)
    assert get_zone(395.0, 190.0) == "Corridor_Cafeteria_Medbay"

    # Outside walls blocked
    assert not is_walkable(10.0, 10.0)
    assert not is_walkable(400.0, 80.0)  # Void between Medbay and Cafeteria

    # Diagonal push against Cafeteria top wall (y=55, min valid y = 69) slides along X
    nx, ny = move_with_sliding(600.0, 70.0, 18.0, -18.0)
    assert is_walkable(nx, ny)
    assert nx > 600.0  # Slid horizontally even though Y hit top wall

    # Line-of-sight wall occlusion checks
    assert has_line_of_sight(210.0, 190.0, 300.0, 190.0)        # Inside Medbay
    assert has_line_of_sight(210.0, 190.0, 500.0, 190.0)        # Down Medbay<->Cafeteria corridor
    assert not has_line_of_sight(320.0, 100.0, 470.0, 100.0)    # Across solid wall above corridor
    assert not has_line_of_sight(210.0, 100.0, 470.0, 190.0)    # Clipped by Medbay doorway corner


def _controlled_engine(impostor: str = "P0") -> MiniAmongUsEngine:
    eng = MiniAmongUsEngine(seed=123)
    eng.impostor_id = impostor
    eng.roles = {p: ("IMPOSTOR" if p == impostor else "CREWMATE") for p in PLAYERS}
    eng.assigned_tasks[impostor] = []
    eng.incomplete_tasks[impostor] = set()
    return eng


def test_realtime_cooldowns_tasks_and_witnesses() -> None:
    """Verify 6s/8s/12s kill cooldowns, 80px kill range, 4 tasks x 5.0s, and 2s vent."""
    eng = _controlled_engine(impostor="P0")

    assert eng.kill_cooldown == INITIAL_KILL_COOLDOWN == 6.0
    assert TASK_DURATION == 5.0
    assert TASK_WIN_THRESHOLD == 12
    assert KILL_RANGE == 80.0
    for p in PLAYERS:
        if p != "P0":
            assert len(eng.assigned_tasks[p]) == TASKS_PER_CREWMATE == 4

    # Place P0 & P1 78px apart (within 80px KILL_RANGE), P2 within vision, P3 & P4 in Storage
    eng.positions["P0"] = (550.0, 190.0)
    eng.positions["P1"] = (628.0, 190.0)
    eng.positions["P2"] = (650.0, 190.0)
    eng.positions["P3"] = (600.0, 610.0)
    eng.positions["P4"] = (620.0, 610.0)

    # At t=0..5.9s kill_cooldown > 0 -> KILL must NOT trigger
    eng.step({"P0": {"kill": True}}, dt=0.1)
    assert "P1" in eng.alive_set
    assert round(eng.kill_cooldown, 1) == 5.9

    # Advance 5.9s more -> kill_cooldown hits 0.0
    for _ in range(59):
        eng.step({}, dt=0.1)
    assert eng.kill_cooldown == 0.0

    # Check 85px out-of-range doesn't kill, then 78px kills cleanly
    eng.positions["P1"] = (635.0, 190.0)
    eng.step({"P0": {"kill": True}}, dt=0.1)
    assert "P1" in eng.alive_set
    eng.positions["P1"] = (628.0, 190.0)

    # Now P0 kills P1 at 78px -> sets POST_KILL_COOLDOWN (12.0s)
    eng.step({"P0": {"kill": True}}, dt=0.1)
    assert "P1" not in eng.alive_set
    assert round(eng.kill_cooldown, 1) == POST_KILL_COOLDOWN == 12.0
    assert len(eng.bodies) == 1
    assert any(e["type"] == "SAW_KILL" and e["player"] == "P2" for e in eng.events_log)

    # Task channeling (5.0s) and movement cancel
    st = eng.assigned_tasks["P3"][0]
    eng.positions["P3"] = (st["x"], st["y"])
    eng.step({"P3": {"interact": True}}, dt=0.1)
    assert eng.is_doing_task["P3"]
    # Moving cancels task
    eng.step({"P3": {"d": True}}, dt=0.1)
    assert not eng.is_doing_task["P3"]
    assert eng.task_progress["P3"] == 0.0

    # Complete task by holding still for 5.0s (50 ticks)
    eng.positions["P3"] = (st["x"], st["y"])
    eng.step({"P3": {"interact": True}}, dt=0.1)
    for _ in range(49):
        eng.step({}, dt=0.1)
    assert not eng.is_doing_task["P3"]
    assert eng.tasks_completed_total == 1

    # Vent test: P0 at Vent_Medbay (135, 265) with P2 nearby (200, 265), vents to Admin (1065, 265)
    v_med = VENTS[0]
    eng.positions["P0"] = (v_med["x"], v_med["y"])
    eng.positions["P2"] = (200.0, 265.0)
    eng.step({"P0": {"vent": True}}, dt=0.1)
    assert eng.positions["P0"] == (VENTS[1]["x"], VENTS[1]["y"])
    assert round(eng.vent_cooldown, 1) == 2.0
    assert any(e["type"] == "SAW_VENT" and e["player"] == "P2" for e in eng.events_log)

    # P1's body is at (628, 190), 28px from button -> interact triggers BODY_REPORT first
    eng.positions["P2"] = EMERGENCY_BUTTON_POS
    eng.step({"P2": {"interact": True}}, dt=0.1)
    assert eng.phase == "MEETING_CHAT"
    assert eng.current_meeting is not None
    assert eng.current_meeting["trigger"] == "BODY_REPORT"

    # Speaking order wrap-around from P2 among alive [P0, P2, P3, P4] -> [P2, P3, P4, P0]
    assert eng.speaking_order == ["P2", "P3", "P4", "P0"]

    # Complete chat + tie vote -> resumes MAP in Cafeteria with 8s kill CD & 15s emergency CD
    for speaker in ["P2", "P3", "P4", "P0"]:
        eng.submit_chat(speaker, "PASS", {})
    eng.step_votes({"P0": "SKIP", "P2": "SKIP", "P3": "SKIP", "P4": "SKIP"})
    assert eng.phase == "MAP"
    assert eng.bodies == []
    assert eng.kill_cooldown == POST_MEETING_KILL_COOLDOWN == 8.0
    assert eng.emergency_cooldown == 15.0

    # Emergency button blocked while emergency_cooldown > 0
    eng.positions["P2"] = EMERGENCY_BUTTON_POS
    eng.step({"P2": {"interact": True}}, dt=0.1)
    assert eng.phase == "MAP"

    # Advance remaining 14.9s -> emergency_cooldown == 0.0 -> CALL_MEETING triggers EMERGENCY
    for _ in range(149):
        eng.step({}, dt=0.1)
    assert eng.emergency_cooldown == 0.0
    eng.step({"P2": {"interact": True}}, dt=0.1)
    assert eng.phase == "MEETING_CHAT"
    assert eng.current_meeting is not None
    assert eng.current_meeting["trigger"] == "EMERGENCY"


def test_continuous_logic_flags() -> None:
    """Verify logic_impossible_travel and logic_witness_contradiction on 10 Hz logs."""
    eng = _controlled_engine(impostor="P0")
    # Simulate 2 seconds in Medbay together, then P0 vents Medbay -> Admin at t=2.1s
    for p in PLAYERS:
        eng.positions[p] = (210.0, 190.0)
    eng.positions["P4"] = (990.0, 190.0)  # P4 sitting in Admin as living witness!

    for _ in range(20):
        eng.step({}, dt=0.1)

    # P0 moves to Vent_Medbay and vents to Admin in 0.1s (impossible on foot!)
    eng.positions["P0"] = (VENTS[0]["x"], VENTS[0]["y"])
    eng.step({"P0": {"vent": True}}, dt=0.1)

    # P1 runs to button after 15s total
    while eng.emergency_cooldown > 0.0:
        eng.step({}, dt=0.1)
    eng.positions["P1"] = EMERGENCY_BUTTON_POS
    eng.step({"P1": {"interact": True}}, dt=0.1)
    assert eng.phase == "MEETING_CHAT"

    # Speaking order starts at P1 -> P2 -> P3 -> P4 -> P0
    win_early = eng.current_meeting["time_windows"][0]  # 0:00-0:10
    eng.submit_chat("P1", "CLAIM_LOC", {"room": "Medbay", "time_window": win_early})  # Truthful
    eng.submit_chat("P2", "CLAIM_LOC", {"room": "Admin", "time_window": win_early})   # Absent in Admin while P4 there + unreachable in 0:00-0:10 from Medbay? Wait: 650/180 = 3.6s <= 10s!
    eng.submit_chat("P3", "CLAIM_LOC", {"room": "Electrical", "time_window": "0:00-0:01"})  # Impossible travel from Cafeteria/Medbay in 1s!
    eng.submit_chat("P4", "PASS", {})
    eng.submit_chat("P0", "PASS", {})
    eng.step_votes({"P0": "SKIP", "P1": "SKIP", "P2": "SKIP", "P3": "SKIP", "P4": "SKIP"})
    eng.phase = "GAME_OVER"
    eng.winner = "IMPOSTOR"
    eng.win_reason = "TIME_LIMIT"

    log = eng.to_game_log()
    m_rows = {r["player_id"]: r for r in extract_meeting_rows(log)}
    assert m_rows["P1"]["logic_impossible_travel"] == 0
    assert m_rows["P1"]["logic_witness_contradiction"] == 0
    assert m_rows["P2"]["logic_witness_contradiction"] == 1  # P4 was in Admin during 0:00-0:10, P2 was not
    assert m_rows["P3"]["logic_impossible_travel"] == 1      # Cannot reach Electrical from Medbay in 0:00-0:01


async def _check_partial_observability_async() -> None:
    """Verify strict partial observability in Engine & FastAPI /api/state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        server.reset_server_state(custom_data_dir=tmpdir)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app),
            base_url="http://test",
        ) as client:
            r = await client.post(
                "/api/lobby/claim",
                json={"player_id": "P0", "client_token": "tok_p0", "name": "Alice"},
            )
            assert r.status_code == 200
            r = await client.post("/api/game/start", json={"seed": 42})
            assert r.status_code == 200

            assert server.engine is not None
            imp = server.engine.impostor_id
            crew = next(p for p in PLAYERS if p != imp)

            # Separate players > 300px apart & place a dead body far away
            server.engine.positions[crew] = ROOM_CENTERS["Medbay"]      # (210, 190)
            server.engine.positions[imp] = ROOM_CENTERS["Electrical"]   # (990, 610)
            server.engine.bodies = [
                {"victim": "P4", "x": 990.0, "y": 610.0, "zone": "Electrical", "t": 12.0}
            ]

            res_crew = await client.get(f"/api/state?player={crew}")
            assert res_crew.status_code == 200
            crew_payload = res_crew.json()
            assert crew_payload["my_role"] == "CREWMATE"
            assert "vents" not in crew_payload["map_geometry"]
            assert "kill_cooldown" not in crew_payload
            assert "IMPOSTOR" not in json.dumps(crew_payload)
            assert all(vp["player_id"] != imp for vp in crew_payload["visible_players"])
            assert crew_payload["visible_bodies"] == []

            res_imp = await client.get(f"/api/state?player={imp}")
            assert res_imp.status_code == 200
            imp_payload = res_imp.json()
            assert imp_payload["my_role"] == "IMPOSTOR"
            assert "vents" in imp_payload["map_geometry"]
            assert "kill_cooldown" in imp_payload
            assert len(imp_payload["visible_bodies"]) == 1

            # Verify wall LOS hides nearby players across solid wall (< 160px Euclidean, no LOS)
            server.engine.positions[crew] = (320.0, 100.0)  # Medbay top-right
            server.engine.positions[imp] = (470.0, 100.0)   # Cafeteria top-left (150px away, wall between!)
            res_wall = await client.get(f"/api/state?player={crew}")
            assert all(vp["player_id"] != imp for vp in res_wall.json()["visible_players"])

        server.reset_server_state(custom_data_dir=server.DEFAULT_DATA_DIR)


def test_sabotage_lights_and_minigame_task_api() -> None:
    """Verify Impostor Sabotage Lights (85px vision, 25s CD), Fix Lights, and mini-game hook."""
    eng = _controlled_engine(impostor="P0")
    eng.positions["P0"] = (210.0, 190.0)
    eng.positions["P1"] = (330.0, 190.0)  # 120px away from P0, clear LOS in Medbay

    # Before sabotage: P1 has 220px vision -> sees P0
    assert eng.get_observation("P1")["vision_radius"] == CREWMATE_VISION == 220.0
    assert any(vp["player_id"] == "P0" for vp in eng.get_observation("P1")["visible_players"])

    # Advance 8.0s past initial sabotage delay, then trigger Lights Out
    for _ in range(80):
        eng.step({}, dt=0.1)
    assert eng.sabotage_cooldown == 0.0
    eng.step({"P0": {"sabotage": True}}, dt=0.1)
    assert eng.lights_out is True
    assert round(eng.sabotage_cooldown, 1) == SABOTAGE_COOLDOWN == 25.0
    assert eng.ticks_log[-1]["lights_out"] is True
    assert any(e["type"] == "SABOTAGE_LIGHTS" for e in eng.events_log)

    # During Lights Out: P1 vision drops to 85px (no longer sees P0 at 120px), P0 keeps 260px!
    obs_p1 = eng.get_observation("P1")
    obs_p0 = eng.get_observation("P0")
    assert obs_p1["vision_radius"] == CREWMATE_LIGHTS_OUT_VISION == 85.0
    assert obs_p0["vision_radius"] == IMPOSTOR_VISION == 260.0
    assert obs_p1["visible_players"] == []
    assert any(vp["player_id"] == "P1" for vp in obs_p0["visible_players"])

    # Fix Lights from Electrical panel
    eng.positions["P1"] = LIGHTS_PANEL_POS
    assert eng.fix_lights("P1") is True
    assert eng.lights_out is False
    assert eng.get_observation("P1")["vision_radius"] == CREWMATE_VISION
    assert any(e["type"] == "FIX_LIGHTS" and e["player"] == "P1" for e in eng.events_log)

    # Human interactive task completion validates proximity
    st = eng.assigned_tasks["P1"][0]
    eng.positions["P1"] = (600.0, 190.0)
    assert eng.complete_task_station("P1", st["station_id"]) is False
    eng.positions["P1"] = (st["x"], st["y"])
    assert eng.complete_task_station("P1", st["station_id"]) is True
    assert eng.tasks_completed_total == 1


def test_20_headless_games_balance_and_csv_export() -> None:
    """Run 20 full 10 Hz headless games, check pacing balance, and verify CSV outputs."""
    n_games = 20
    t0 = time.perf_counter()
    games = [simulate_game(seed=1000 + i) for i in range(n_games)]
    elapsed = time.perf_counter() - t0
    print(f"[BENCHMARK] Simulated {n_games} full 10 Hz 2D games in {elapsed:.4f}s")

    total_ticks = 0
    total_meetings = 0
    total_kills = 0
    crew_wins = 0
    imp_wins = 0
    durations: list[float] = []

    for g in games:
        assert g["winner"] in ("CREWMATES", "IMPOSTOR")
        if g["winner"] == "CREWMATES":
            crew_wins += 1
        else:
            imp_wins += 1

        assert len(g["ticks"]) > 0
        durations.append(float(g["total_time"]))
        total_ticks += len(g["ticks"])
        total_meetings += len(g["meetings"])
        total_kills += sum(1 for e in g["events"] if e["type"] == "KILL")

        for tick in g["ticks"]:
            for pid, pdata in tick["players"].items():
                assert is_walkable(pdata["x"], pdata["y"]), (
                    f"Out-of-bounds position for {pid} at t={tick['t']}: "
                    f"({pdata['x']}, {pdata['y']})"
                )
                assert pdata["zone"] in ROOM_AABBS or pdata["zone"].startswith("Corridor_")

    avg_duration = sum(durations) / n_games
    crew_rate = (crew_wins / n_games) * 100.0
    imp_rate = (imp_wins / n_games) * 100.0
    avg_kills = total_kills / n_games

    print(
        f"[BALANCE] Avg duration: {avg_duration:.1f}s (min {min(durations):.1f}s, max {max(durations):.1f}s) | "
        f"Crewmate win rate: {crew_rate:.1f}% ({crew_wins}/{n_games}) | "
        f"Impostor win rate: {imp_rate:.1f}% ({imp_wins}/{n_games}) | "
        f"Avg kills/game: {avg_kills:.2f}"
    )

    assert 35.0 <= avg_duration <= 100.0, (
        f"Average game duration {avg_duration:.1f}s outside 35-100s target"
    )
    assert avg_kills > 0.8, f"Average kills/game {avg_kills:.2f} <= 0.8"
    assert total_meetings > 0, "Expected meetings across 20 simulated games"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        jsonl_path = tmp / "games_log.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for g in games:
                f.write(json.dumps(g, separators=(",", ":")) + "\n")

        t_cnt, m_cnt = export_all(jsonl_path, tmp)
        assert t_cnt > 0 and m_cnt > 0

        traj_csv = tmp / "keystroke_trajectories.csv"
        meet_csv = tmp / "meeting_features.csv"
        assert traj_csv.exists() and meet_csv.exists()

        with traj_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
            assert tuple(rows[0]) == KEYSTROKE_TRAJECTORY_COLUMNS
            assert len(rows) - 1 == t_cnt

        with meet_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
            assert tuple(rows[0]) == MEETING_FEATURE_COLUMNS
            assert len(rows) - 1 == m_cnt

        print(
            f"[CSV CHECK] Temp export verified: {t_cnt} trajectory rows, "
            f"{m_cnt} meeting rows across {total_ticks} ticks & {total_meetings} meetings."
        )


def main() -> None:
    test_engine_stdlib_only()
    print("[PASS] engine.py imports 100% Python stdlib only.")

    test_geometry_collision_and_zones()
    print("[PASS] 2D AABB geometry, diagonal wall sliding, and get_zone() verified.")

    test_realtime_cooldowns_tasks_and_witnesses()
    print("[PASS] Real-time cooldowns (6s/8s/12s kill, 80px range, 5s task, 2s vent, 15s meeting) & witnesses verified.")

    test_continuous_logic_flags()
    print("[PASS] Continuous Logical Agent flags verified.")

    asyncio.run(_check_partial_observability_async())
    print("[PASS] Strict partial observability, LOS wall occlusion, & FastAPI /api/state isolation verified.")

    test_sabotage_lights_and_minigame_task_api()
    print("[PASS] Sabotage Lights (85px/260px vision, 25s CD), Fix Lights, & mini-game completion verified.")

    test_20_headless_games_balance_and_csv_export()
    print("[PASS] All 20 headless balance & CSV export checks passed!")


if __name__ == "__main__":
    main()
