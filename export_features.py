"""Deterministic feature extraction pipeline for real-time 10 Hz Mini-Among Us.

Reads raw ground-truth JSONL records from `data/games_log.jsonl` and generates:
1. `data/keystroke_trajectories.csv` (1 row per alive player per 10 Hz tick)
2. `data/meeting_features.csv` (1 row per alive player per meeting)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from engine import (
    EMERGENCY_BUTTON_POS,
    PLAYER_SPEED,
    PLAYERS,
    ROOM_AABBS,
    ROOMS,
    SPAWN_POINTS,
    min_walk_dist_to_room,
    parse_time_window,
)

KEYSTROKE_TRAJECTORY_COLUMNS: tuple[str, ...] = (
    "game_id",
    "t",
    "player_id",
    "role",
    "is_human",
    "x",
    "y",
    "vx",
    "vy",
    "zone",
    "key_w",
    "key_a",
    "key_s",
    "key_d",
    "action",
    "num_visible_players",
    "body_visible",
    "dist_to_nearest_player",
    "near_task",
    "kill_cooldown",
    "tasks_completed_total",
    "alive_count",
    "lights_out",
)

MEETING_FEATURE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "meeting_id",
    "player_id",
    "is_impostor",
    "is_human",
    "round_duration_sec",
    "time_alone_ratio",
    "unique_witnesses_count",
    "avg_speed",
    "idle_ratio",
    "direction_change_rate",
    "unique_rooms_visited",
    "min_dist_to_body_last_seen",
    "reported_body",
    "chat_type",
    "chat_order",
    "vouches_received",
    "accusations_received",
    "chat_rt_ms",
    "vote_rt_ms",
    "logic_impossible_travel",
    "logic_witness_contradiction",
    "voted_for_impostor",
)


def _format_rt_ms(val: Any, is_human: int) -> str:
    if not is_human or val is None:
        return "NaN"
    return f"{float(val):.1f}"


def extract_keystroke_rows(game: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract 1 row per alive player per 10 Hz tick from one game log."""
    rows: list[dict[str, Any]] = []
    game_id = game["game_id"]
    players_meta = game["players"]

    for tick in game.get("ticks", []):
        t_val = float(tick["t"])
        tasks_total = int(tick["tasks_completed_total"])
        p_map: dict[str, dict[str, Any]] = tick.get("players", {})
        alive_pids = [p for p in PLAYERS if p in p_map]
        alive_count = len(alive_pids)

        for pid in alive_pids:
            pdata = p_map[pid]
            meta = players_meta[pid]
            px = float(pdata["x"])
            py = float(pdata["y"])

            nearest_d = -1.0
            if alive_count > 1:
                min_d = float("inf")
                for other in alive_pids:
                    if other == pid:
                        continue
                    ox = float(p_map[other]["x"])
                    oy = float(p_map[other]["y"])
                    d = math.hypot(px - ox, py - oy)
                    if d < min_d:
                        min_d = d
                nearest_d = round(min_d, 1)

            keys = pdata.get("keys", {})
            rows.append(
                {
                    "game_id": game_id,
                    "t": round(t_val, 1),
                    "player_id": pid,
                    "role": meta["role"],
                    "is_human": 1 if meta["is_human"] else 0,
                    "x": round(px, 1),
                    "y": round(py, 1),
                    "vx": round(float(pdata["vx"]), 1),
                    "vy": round(float(pdata["vy"]), 1),
                    "zone": pdata["zone"],
                    "key_w": int(keys.get("w", 0)),
                    "key_a": int(keys.get("a", 0)),
                    "key_s": int(keys.get("s", 0)),
                    "key_d": int(keys.get("d", 0)),
                    "action": pdata.get("action", "NONE"),
                    "num_visible_players": len(pdata.get("visible_players", [])),
                    "body_visible": 1 if pdata.get("visible_bodies") else 0,
                    "dist_to_nearest_player": nearest_d,
                    "near_task": 1 if pdata.get("near_task") else 0,
                    "kill_cooldown": round(float(pdata.get("kill_cooldown", 0.0)), 2),
                    "tasks_completed_total": tasks_total,
                    "alive_count": alive_count,
                    "lights_out": 1 if tick.get("lights_out", False) else 0,
                }
            )
    return rows


def _compute_continuous_logic_flags(
    pid: str,
    chat_entry: dict[str, Any],
    meeting: dict[str, Any],
    round_ticks: list[dict[str, Any]],
) -> tuple[int, int]:
    """Compute (logic_impossible_travel, logic_witness_contradiction) over 10 Hz round ticks."""
    ctype = chat_entry["type"]
    if ctype not in ("CLAIM_LOC", "VOUCH"):
        return (0, 0)

    args = chat_entry.get("args", {})
    claimed_room = str(args.get("room", ""))
    if claimed_room not in ROOM_AABBS:
        return (1, 0)

    alive_at_meeting = set(meeting["alive_at_meeting"])
    r_start = float(meeting["round_start_t"])
    t_called = float(meeting["t_called"])

    if ctype == "CLAIM_LOC":
        win = parse_time_window(str(args.get("time_window", "")))
        if win is None:
            return (1, 0)
        w_start, w_end = win
    else:
        w_start, w_end = r_start, t_called

    # 1. Impossible travel against witnessed position anchors
    sx, sy = SPAWN_POINTS[pid]
    anchors: list[tuple[float, float, float]] = [(r_start, sx, sy)]
    for tick in round_ticks:
        p_map = tick.get("players", {})
        if pid not in p_map:
            continue
        witnessed = any(
            w in p_map and w != pid and pid in p_map[w].get("visible_players", [])
            for w in alive_at_meeting
        )
        if witnessed:
            anchors.append((float(tick["t"]), float(p_map[pid]["x"]), float(p_map[pid]["y"])))

    if meeting["caller"] == pid:
        if meeting["trigger"] == "BODY_REPORT" and meeting.get("body_x") is not None:
            anchors.append((t_called, float(meeting["body_x"]), float(meeting["body_y"])))
        elif meeting["trigger"] == "EMERGENCY":
            anchors.append((t_called, EMERGENCY_BUTTON_POS[0], EMERGENCY_BUTTON_POS[1]))

    # Sample candidate timestamps across [w_start, w_end]
    candidates: list[float] = [w_start, w_end]
    cur = w_start + 0.5
    while cur < w_end:
        candidates.append(cur)
        cur += 0.5

    anchor_dists = [
        (t_anc, min_walk_dist_to_room(ax, ay, claimed_room))
        for t_anc, ax, ay in anchors
    ]
    can_reach = False
    for tau in candidates:
        feasible = True
        for t_anc, dist in anchor_dists:
            max_reach = PLAYER_SPEED * (abs(tau - t_anc) + 0.25)
            if dist > max_reach:
                feasible = False
                break
        if feasible:
            can_reach = True
            break
    impossible_travel = 0 if can_reach else 1

    # 2. Witness contradiction inside [w_start, w_end]
    window_ticks = [
        tick
        for tick in round_ticks
        if w_start - 0.1 <= float(tick["t"]) <= w_end + 0.1
    ]
    pid_in_room = any(
        pid in tick["players"] and tick["players"][pid]["zone"] == claimed_room
        for tick in window_ticks
    )
    witnesses_in_room = [
        w
        for w in alive_at_meeting
        if w != pid
        and any(
            w in tick["players"] and tick["players"][w]["zone"] == claimed_room
            for tick in window_ticks
        )
    ]

    witness_contradiction = 0
    if witnesses_in_room and not pid_in_room:
        witness_contradiction = 1
    elif ctype == "VOUCH":
        v_target = str(args.get("target", ""))
        target_in_room = any(
            v_target in tick["players"] and tick["players"][v_target]["zone"] == claimed_room
            for tick in window_ticks
        )
        other_witnesses = [w for w in witnesses_in_room if w != v_target]
        if other_witnesses and not target_in_room:
            witness_contradiction = 1

    return (impossible_travel, witness_contradiction)


def extract_meeting_rows(game: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract 1 feature row per alive player per meeting with 10 Hz movement stats."""
    rows: list[dict[str, Any]] = []
    game_id = game["game_id"]
    players_meta = game["players"]
    ticks: list[dict[str, Any]] = game.get("ticks", [])
    events: list[dict[str, Any]] = game.get("events", [])
    meetings: list[dict[str, Any]] = game.get("meetings", [])

    impostor_id = next(
        p for p, info in players_meta.items() if info["role"] == "IMPOSTOR"
    )

    ticks_by_round: dict[int, list[dict[str, Any]]] = {}
    for tick in ticks:
        ticks_by_round.setdefault(int(tick["round"]), []).append(tick)

    events_by_round: dict[int, list[dict[str, Any]]] = {}
    for ev in events:
        events_by_round.setdefault(int(ev["round"]), []).append(ev)

    for m in meetings:
        meeting_id = int(m["meeting_id"])
        round_num = int(m["round"])
        r_start = float(m["round_start_t"])
        t_called = float(m["t_called"])
        round_duration = round(max(0.0, t_called - r_start), 1)
        round_ticks = ticks_by_round.get(round_num, [])
        round_events = events_by_round.get(round_num, [])

        quick_chat = m.get("quick_chat", [])
        chat_by_player = {c["player"]: c for c in quick_chat}
        votes = m.get("votes", {})
        is_body_report = (
            m["trigger"] == "BODY_REPORT"
            and m.get("body_x") is not None
            and m.get("body_y") is not None
        )
        bx = float(m["body_x"]) if is_body_report else 0.0
        by = float(m["body_y"]) if is_body_report else 0.0

        for pid in m["alive_at_meeting"]:
            meta = players_meta[pid]
            is_imp = 1 if meta["role"] == "IMPOSTOR" else 0
            is_hum = 1 if meta["is_human"] else 0

            alone_ticks = 0
            idle_ticks = 0
            speed_sum = 0.0
            dir_changes = 0
            dir_transitions = 0
            prev_keys: tuple[int, int, int, int] | None = None
            rooms_visited: set[str] = set()
            witnesses: set[str] = set()
            last_wit_x, last_wit_y = SPAWN_POINTS[pid]
            p_tick_count = 0

            for tick in round_ticks:
                p_map = tick.get("players", {})
                pdata = p_map.get(pid)
                if pdata is None:
                    continue
                p_tick_count += 1

                vis_list = pdata.get("visible_players", [])
                if len(vis_list) == 0:
                    alone_ticks += 1

                # Who saw `pid` on this tick
                saw_pid = False
                for other, odata in p_map.items():
                    if other != pid and pid in odata.get("visible_players", []):
                        witnesses.add(other)
                        saw_pid = True
                if saw_pid:
                    last_wit_x = float(pdata["x"])
                    last_wit_y = float(pdata["y"])

                speed_sum += math.hypot(float(pdata["vx"]), float(pdata["vy"]))

                keys = pdata.get("keys", {})
                k_tuple = (
                    int(keys.get("w", 0)),
                    int(keys.get("a", 0)),
                    int(keys.get("s", 0)),
                    int(keys.get("d", 0)),
                )
                doing_task = bool(pdata.get("is_doing_task", False))
                if k_tuple == (0, 0, 0, 0) and not doing_task:
                    idle_ticks += 1

                if prev_keys is not None and not doing_task:
                    dir_transitions += 1
                    if k_tuple != prev_keys:
                        dir_changes += 1
                prev_keys = k_tuple

                zone = pdata["zone"]
                if zone in ROOMS:
                    rooms_visited.add(zone)

            # Also count explicit kill/vent witnesses from events
            for ev in round_events:
                if ev.get("type") in ("KILL", "VENT") and ev.get("player") == pid:
                    witnesses.update(ev.get("witnesses", []))
            witnesses.discard(pid)

            time_alone_ratio = (
                round(alone_ticks / p_tick_count, 6) if p_tick_count > 0 else 0.0
            )
            avg_speed = (
                round(speed_sum / p_tick_count, 2) if p_tick_count > 0 else 0.0
            )
            idle_ratio = (
                round(idle_ticks / p_tick_count, 6) if p_tick_count > 0 else 0.0
            )
            direction_change_rate = (
                round(dir_changes / dir_transitions, 6)
                if dir_transitions > 0
                else 0.0
            )
            min_dist_to_body = (
                round(math.hypot(last_wit_x - bx, last_wit_y - by), 1)
                if is_body_report
                else -1.0
            )

            chat_entry = chat_by_player.get(
                pid, {"type": "PASS", "order": -1, "args": {}, "rt_ms": None}
            )
            vouches_received = sum(
                1
                for c in quick_chat
                if c["player"] != pid
                and c["type"] == "VOUCH"
                and c.get("args", {}).get("target") == pid
            )
            accusations_received = sum(
                1
                for c in quick_chat
                if c["player"] != pid
                and c["type"] == "ACCUSE"
                and c.get("args", {}).get("target") == pid
            )
            vote_entry = votes.get(pid, {"target": "SKIP", "rt_ms": None})

            imp_travel, wit_contra = _compute_continuous_logic_flags(
                pid=pid,
                chat_entry=chat_entry,
                meeting=m,
                round_ticks=round_ticks,
            )

            rows.append(
                {
                    "game_id": game_id,
                    "meeting_id": meeting_id,
                    "player_id": pid,
                    "is_impostor": is_imp,
                    "is_human": is_hum,
                    "round_duration_sec": round_duration,
                    "time_alone_ratio": time_alone_ratio,
                    "unique_witnesses_count": len(witnesses),
                    "avg_speed": avg_speed,
                    "idle_ratio": idle_ratio,
                    "direction_change_rate": direction_change_rate,
                    "unique_rooms_visited": len(rooms_visited),
                    "min_dist_to_body_last_seen": min_dist_to_body,
                    "reported_body": 1 if m["caller"] == pid else 0,
                    "chat_type": chat_entry["type"],
                    "chat_order": int(chat_entry["order"]),
                    "vouches_received": vouches_received,
                    "accusations_received": accusations_received,
                    "chat_rt_ms": _format_rt_ms(chat_entry.get("rt_ms"), is_hum),
                    "vote_rt_ms": _format_rt_ms(vote_entry.get("rt_ms"), is_hum),
                    "logic_impossible_travel": imp_travel,
                    "logic_witness_contradiction": wit_contra,
                    "voted_for_impostor": 1 if vote_entry["target"] == impostor_id else 0,
                }
            )
    return rows


def export_all(
    jsonl_path: str | Path = "data/games_log.jsonl",
    out_dir: str | Path = "data",
) -> tuple[int, int]:
    """Parse `jsonl_path` and write `keystroke_trajectories.csv` & `meeting_features.csv`."""
    src = Path(jsonl_path)
    dst_dir = Path(out_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    traj_rows: list[dict[str, Any]] = []
    meeting_rows: list[dict[str, Any]] = []

    if src.exists():
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                game = json.loads(stripped)
                traj_rows.extend(extract_keystroke_rows(game))
                meeting_rows.extend(extract_meeting_rows(game))

    traj_csv = dst_dir / "keystroke_trajectories.csv"
    with traj_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KEYSTROKE_TRAJECTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(traj_rows)

    meeting_csv = dst_dir / "meeting_features.csv"
    with meeting_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MEETING_FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(meeting_rows)

    return (len(traj_rows), len(meeting_rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Mini-Among Us 10 Hz JSONL logs to Colab-ready CSVs."
    )
    parser.add_argument(
        "--input",
        default="data/games_log.jsonl",
        help="Path to raw JSONL log file (default: data/games_log.jsonl)",
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Directory for output CSVs (default: data)",
    )
    args = parser.parse_args()
    t_count, m_count = export_all(args.input, args.out_dir)
    print(
        f"Exported {t_count} rows to {Path(args.out_dir) / 'keystroke_trajectories.csv'} "
        f"and {m_count} rows to {Path(args.out_dir) / 'meeting_features.csv'}."
    )


if __name__ == "__main__":
    main()
