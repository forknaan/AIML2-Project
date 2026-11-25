"""FastAPI + WebSocket real-time 10 Hz multiplayer server for Mini-Among Us.

Runs a deterministic 10 Hz simulation loop (`dt = 0.1s`), accepts held WASD
keystrokes and one-shot triggers over WebSockets, and broadcasts strict
per-player partial-observability snapshots (never leaking off-vision positions,
hidden roles, or vents to Crewmates).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

from engine import (
    DEFAULT_DT,
    MiniAmongUsEngine,
    PLAYERS,
    choose_bot_chat,
    choose_bot_step_input,
    choose_bot_vote,
)
import export_features

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = BASE_DIR / "templates"
ALLOWED_DOWNLOADS = {
    "games_log.jsonl",
    "keystroke_trajectories.csv",
    "meeting_features.csv",
}
SERVER_PORT: int = 5000

data_dir: Path = DEFAULT_DATA_DIR
lobby_slots: dict[str, dict[str, Any]] = {
    p: {"claimed_by": None, "name": "", "force_bot": False} for p in PLAYERS
}
fill_bots: bool = True
engine: MiniAmongUsEngine | None = None
game_logged: bool = False
held_keys: dict[str, dict[str, bool]] = {
    p: {"w": False, "a": False, "s": False, "d": False} for p in PLAYERS
}
pending_triggers: dict[str, dict[str, bool]] = {
    p: {"interact": False, "kill": False, "vent": False, "sabotage": False}
    for p in PLAYERS
}
bot_memories: dict[str, dict[str, Any]] = {p: {} for p in PLAYERS}
speaker_key: tuple[str, int, int] | None = None
speaker_started_at: float | None = None
ws_clients: dict[WebSocket, tuple[str, str]] = {}
_loop_task: asyncio.Task[None] | None = None


def set_data_dir(path: str | Path) -> None:
    global data_dir
    data_dir = Path(path)
    data_dir.mkdir(parents=True, exist_ok=True)


def get_games_log_path() -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "games_log.jsonl"


def count_logged_games() -> int:
    log_path = get_games_log_path()
    if not log_path.exists():
        return 0
    with log_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def get_local_ip() -> str:
    """Detect host laptop LAN IP for Wi-Fi / hotspot multiplayer."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(0.5)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        if ip:
            return ip
    except OSError:
        pass
    finally:
        sock.close()
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    return "127.0.0.1"


def _clear_inputs() -> None:
    for p in PLAYERS:
        held_keys[p] = {"w": False, "a": False, "s": False, "d": False}
        pending_triggers[p] = {
            "interact": False,
            "kill": False,
            "vent": False,
            "sabotage": False,
        }


def reset_server_state(custom_data_dir: str | Path | None = None) -> None:
    """Reset lobby & game globals (primarily for deterministic tests)."""
    global engine, game_logged, fill_bots, speaker_key, speaker_started_at
    if custom_data_dir is not None:
        set_data_dir(custom_data_dir)
    for p in PLAYERS:
        lobby_slots[p] = {"claimed_by": None, "name": "", "force_bot": False}
        bot_memories[p] = {}
    _clear_inputs()
    fill_bots = True
    engine = None
    game_logged = False
    speaker_key = None
    speaker_started_at = None


def _lobby_payload(token: str = "") -> dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)
    slots_out = {}
    my_slot = None
    for p in PLAYERS:
        slot = lobby_slots[p]
        slots_out[p] = {
            "claimed": slot["claimed_by"] is not None,
            "name": slot["name"],
            "force_bot": bool(slot["force_bot"]),
        }
        if token and slot["claimed_by"] == token:
            my_slot = p
    lan_ip = get_local_ip()
    payload: dict[str, Any] = {
        "slots": slots_out,
        "fill_bots": fill_bots,
        "game_active": engine is not None,
        "game_phase": engine.phase if engine is not None else None,
        "lan_ip": lan_ip,
        "lan_url": f"http://{lan_ip}:{SERVER_PORT}",
        "games_logged_count": count_logged_games(),
        "csv_ready": {
            "keystroke_trajectories": (data_dir / "keystroke_trajectories.csv").exists(),
            "meeting_features": (data_dir / "meeting_features.csv").exists(),
        },
    }
    if token:
        payload["my_slot"] = my_slot
    return payload


def _update_speaker_timer() -> None:
    global speaker_key, speaker_started_at
    if (
        engine is not None
        and engine.phase == "MEETING_CHAT"
        and engine.current_meeting is not None
    ):
        key = (
            engine.game_id,
            int(engine.current_meeting["meeting_id"]),
            int(engine.speaker_idx),
        )
        if key != speaker_key:
            speaker_key = key
            speaker_started_at = time.monotonic()
    else:
        speaker_key = None
        speaker_started_at = None


def _flush_completed_game() -> None:
    global game_logged
    if engine is None or engine.phase != "GAME_OVER" or game_logged:
        return
    log_record = engine.to_game_log()
    log_path = get_games_log_path()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_record, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
    game_logged = True


def _advance_meeting_bots() -> None:
    """Auto-submit bot Quick Chat statements and votes immediately."""
    if engine is None:
        return
    safety = 0
    while engine.phase in ("MEETING_CHAT", "MEETING_VOTE") and safety < 100:
        safety += 1
        if engine.phase == "MEETING_CHAT":
            speaker = engine.current_speaker()
            if speaker is not None and not engine.is_human[speaker]:
                obs = engine.get_observation(speaker)
                ctype, cargs = choose_bot_chat(obs, engine.rng)
                engine.submit_chat(speaker, ctype, cargs, rt_ms=0)
            else:
                break
        elif engine.phase == "MEETING_VOTE":
            assert engine.current_meeting is not None
            progressed = False
            for p in engine.current_meeting["alive_at_meeting"]:
                if not engine.is_human[p] and p not in engine.pending_votes:
                    obs = engine.get_observation(p)
                    target = choose_bot_vote(obs, engine.rng)
                    resolved = engine.submit_vote(p, target, rt_ms=0)
                    progressed = True
                    if resolved or engine.phase != "MEETING_VOTE":
                        break
            if engine.phase == "MEETING_VOTE":
                if not progressed or not any(
                    not engine.is_human[p] and p not in engine.pending_votes
                    for p in engine.current_meeting["alive_at_meeting"]
                ):
                    break
    _update_speaker_timer()
    _flush_completed_game()


def _build_state_payload(player: str, token: str = "") -> dict[str, Any]:
    if engine is None:
        return {
            "in_lobby": True,
            "my_player": player,
            "lobby": _lobby_payload(token=token),
        }

    obs = engine.get_observation(player)
    player_names = {
        p: (
            lobby_slots[p]["name"]
            if lobby_slots[p]["claimed_by"] is not None and lobby_slots[p]["name"]
            else (p if engine.is_human[p] else f"{p} (Bot)")
        )
        for p in PLAYERS
    }
    elapsed = 0.0
    if engine.phase == "MEETING_CHAT" and speaker_started_at is not None:
        elapsed = max(0.0, time.monotonic() - speaker_started_at)

    obs["in_lobby"] = False
    obs["player_names"] = player_names
    obs["is_human_map"] = dict(engine.is_human)
    obs["speaker_elapsed_sec"] = round(elapsed, 1)
    obs["speaker_timeout_sec"] = 20
    obs["games_logged_count"] = count_logged_games()
    return obs


async def _broadcast_all() -> None:
    """Broadcast personalized partial-observability payloads to all connected WebSockets."""
    if not ws_clients:
        return
    stale: list[WebSocket] = []
    for ws, (pid, tok) in list(ws_clients.items()):
        try:
            payload = _build_state_payload(pid, token=tok)
            await ws.send_json(payload)
        except Exception:
            stale.append(ws)
    for ws in stale:
        entry = ws_clients.pop(ws, None)
        if entry is not None and entry[0] in PLAYERS:
            held_keys[entry[0]] = {"w": False, "a": False, "s": False, "d": False}


def _step_server_tick() -> bool:
    """Run one 10 Hz physics/AI tick if in MAP phase. Returns True if clients need broadcast."""
    if engine is None:
        return False

    if engine.phase == "MAP":
        inputs_dict: dict[str, dict[str, bool]] = {}
        for p in engine.alive_players():
            if engine.is_human[p]:
                hk = held_keys[p]
                pt = pending_triggers[p]
                inputs_dict[p] = {
                    "w": hk["w"],
                    "a": hk["a"],
                    "s": hk["s"],
                    "d": hk["d"],
                    "interact": pt["interact"],
                    "kill": pt["kill"],
                    "vent": pt["vent"],
                    "sabotage": pt["sabotage"],
                }
                pending_triggers[p] = {
                    "interact": False,
                    "kill": False,
                    "vent": False,
                    "sabotage": False,
                }
            else:
                inputs_dict[p] = choose_bot_step_input(
                    engine.get_observation(p), bot_memories[p], engine.rng
                )

        engine.step(inputs_dict, dt=DEFAULT_DT)
        if engine.phase != "MAP":
            _clear_inputs()
            _advance_meeting_bots()
        return True

    if engine.phase in ("MEETING_CHAT", "MEETING_VOTE"):
        prev_phase = engine.phase
        _advance_meeting_bots()
        if engine.phase != prev_phase:
            return True
        return True  # Broadcast soft speaker timer countdown smoothly

    return False


async def _tick_loop() -> None:
    """Background 10 Hz real-time simulation loop."""
    next_tick = time.monotonic()
    while True:
        now = time.monotonic()
        if now < next_tick:
            await asyncio.sleep(next_tick - now)
        next_tick = max(time.monotonic(), next_tick + DEFAULT_DT)

        should_broadcast = _step_server_tick()
        if should_broadcast:
            await _broadcast_all()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _loop_task
    _loop_task = asyncio.create_task(_tick_loop())
    try:
        yield
    finally:
        if _loop_task is not None:
            _loop_task.cancel()
            try:
                await _loop_task
            except asyncio.CancelledError:
                pass
            _loop_task = None


app = FastAPI(title="Mini-Among Us 2D", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/api/lobby")
async def api_lobby(token: str = Query(default="")) -> JSONResponse:
    return JSONResponse(_lobby_payload(token=token))


@app.post("/api/lobby/claim")
async def api_lobby_claim(request: Request) -> JSONResponse:
    data = await request.json()
    player_id = str(data.get("player_id", ""))
    token = str(data.get("client_token", "")).strip()
    name = str(data.get("name", "")).strip()[:20]

    if player_id not in PLAYERS or not token:
        raise HTTPException(status_code=400, detail="Invalid player_id or client_token")
    if engine is not None and engine.phase != "GAME_OVER":
        raise HTTPException(status_code=400, detail="Game already in progress")

    target_slot = lobby_slots[player_id]
    if target_slot["claimed_by"] is not None and target_slot["claimed_by"] != token:
        raise HTTPException(status_code=409, detail=f"{player_id} is already claimed")

    for p in PLAYERS:
        if p != player_id and lobby_slots[p]["claimed_by"] == token:
            lobby_slots[p]["claimed_by"] = None
            lobby_slots[p]["name"] = ""

    target_slot["claimed_by"] = token
    target_slot["name"] = name or player_id
    target_slot["force_bot"] = False
    return JSONResponse(_lobby_payload(token=token))


@app.post("/api/lobby/release")
async def api_lobby_release(request: Request) -> JSONResponse:
    data = await request.json()
    player_id = str(data.get("player_id", ""))
    token = str(data.get("client_token", "")).strip()

    if player_id not in PLAYERS or not token:
        raise HTTPException(status_code=400, detail="Invalid player_id or client_token")
    if engine is not None and engine.phase != "GAME_OVER":
        raise HTTPException(status_code=400, detail="Cannot release slot mid-game")

    slot = lobby_slots[player_id]
    if slot["claimed_by"] == token:
        slot["claimed_by"] = None
        slot["name"] = ""
    return JSONResponse(_lobby_payload(token=token))


@app.post("/api/lobby/config")
async def api_lobby_config(request: Request) -> JSONResponse:
    global fill_bots
    data = await request.json()
    if "fill_bots" in data:
        fill_bots = bool(data["fill_bots"])
    toggle_slot = data.get("toggle_bot_slot")
    if toggle_slot in PLAYERS:
        slot = lobby_slots[toggle_slot]
        if slot["claimed_by"] is None:
            slot["force_bot"] = not bool(slot["force_bot"])
    return JSONResponse(_lobby_payload())


@app.post("/api/game/start")
async def api_game_start(request: Request) -> JSONResponse:
    global engine, game_logged, speaker_key, speaker_started_at
    try:
        data = await request.json()
    except Exception:
        data = {}
    seed = data.get("seed") if isinstance(data, dict) else None

    is_human: dict[str, bool] = {}
    for p in PLAYERS:
        slot = lobby_slots[p]
        if slot["claimed_by"] is not None:
            is_human[p] = True
        elif fill_bots or slot["force_bot"]:
            is_human[p] = False
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Slot {p} is empty. Enable 'Fill empty slots with Bots' "
                    "or mark empty slots as Bot."
                ),
            )

    _clear_inputs()
    for p in PLAYERS:
        bot_memories[p] = {}
    engine = MiniAmongUsEngine(
        seed=int(seed) if seed is not None else None,
        is_human=is_human,
    )
    game_logged = False
    speaker_key = None
    speaker_started_at = None

    # If 5-bot game in test/demo, let background loop or immediate fast-forward handle it
    if not any(is_human.values()) and data.get("run_to_completion", False):
        while engine.phase != "GAME_OVER":
            _step_server_tick()
    await _broadcast_all()
    return JSONResponse({"ok": True, "game_id": engine.game_id, "phase": engine.phase})


@app.post("/api/game/lobby")
async def api_game_lobby() -> JSONResponse:
    global engine, game_logged, speaker_key, speaker_started_at
    engine = None
    game_logged = False
    speaker_key = None
    speaker_started_at = None
    _clear_inputs()
    await _broadcast_all()
    return JSONResponse({"ok": True, "lobby": _lobby_payload()})


@app.get("/api/state")
async def api_state(
    player: str = Query(default=""),
    token: str = Query(default=""),
) -> JSONResponse:
    if player not in PLAYERS:
        raise HTTPException(status_code=400, detail="Missing or invalid ?player=P0..P4")
    return JSONResponse(_build_state_payload(player, token=token))


@app.post("/api/chat")
async def api_chat(request: Request) -> JSONResponse:
    data = await request.json()
    player_id = str(data.get("player_id", ""))
    chat_type = str(data.get("type", ""))
    args = data.get("args") or {}
    rt_ms = data.get("rt_ms", 0)

    if player_id not in PLAYERS or engine is None:
        raise HTTPException(status_code=400, detail="Invalid player_id or no active game")
    try:
        engine.submit_chat(player_id, chat_type, args=args, rt_ms=rt_ms)
        _advance_meeting_bots()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await _broadcast_all()
    return JSONResponse({"ok": True, "phase": engine.phase})


@app.post("/api/vote")
async def api_vote(request: Request) -> JSONResponse:
    data = await request.json()
    player_id = str(data.get("player_id", ""))
    target = str(data.get("target", ""))
    rt_ms = data.get("rt_ms", 0)

    if player_id not in PLAYERS or engine is None:
        raise HTTPException(status_code=400, detail="Invalid player_id or no active game")
    try:
        engine.submit_vote(player_id, target, rt_ms=rt_ms)
        _advance_meeting_bots()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await _broadcast_all()
    return JSONResponse({"ok": True, "phase": engine.phase})


@app.post("/api/export")
async def api_export() -> JSONResponse:
    t_rows, m_rows = export_features.export_all(
        jsonl_path=get_games_log_path(),
        out_dir=data_dir,
    )
    return JSONResponse(
        {
            "ok": True,
            "games_logged_count": count_logged_games(),
            "trajectory_rows": t_rows,
            "meeting_rows": m_rows,
        }
    )


@app.get("/api/download/{filename}")
async def api_download(filename: str) -> FileResponse:
    if filename not in ALLOWED_DOWNLOADS:
        raise HTTPException(status_code=404, detail="File not downloadable")
    file_path = data_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} does not exist yet")
    return FileResponse(path=str(file_path), filename=filename)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    player = websocket.query_params.get("player", "")
    token = websocket.query_params.get("token", "")
    if player not in PLAYERS:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    ws_clients[websocket] = (player, token)
    try:
        await websocket.send_json(_build_state_payload(player, token=token))
        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")

            if mtype == "keys":
                held_keys[player] = {
                    "w": bool(msg.get("w", False)),
                    "a": bool(msg.get("a", False)),
                    "s": bool(msg.get("s", False)),
                    "d": bool(msg.get("d", False)),
                }
            elif mtype == "trigger":
                act = str(msg.get("action", ""))
                if act in ("interact", "kill", "vent", "sabotage"):
                    pending_triggers[player][act] = True
            elif mtype == "task_complete" and engine is not None:
                sid = str(msg.get("station_id", ""))
                if engine.complete_task_station(player, sid):
                    _flush_completed_game()
                    await _broadcast_all()
            elif mtype == "cancel_task" and engine is not None:
                engine.cancel_task(player)
                await _broadcast_all()
            elif mtype == "fix_lights" and engine is not None:
                if engine.fix_lights(player):
                    await _broadcast_all()
            elif mtype == "chat" and engine is not None:
                try:
                    engine.submit_chat(
                        player,
                        str(msg.get("chat_type", "")),
                        args=msg.get("args") or {},
                        rt_ms=msg.get("rt_ms", 0),
                    )
                    _advance_meeting_bots()
                    await _broadcast_all()
                except ValueError as exc:
                    await websocket.send_json({"error": str(exc)})
            elif mtype == "vote" and engine is not None:
                try:
                    engine.submit_vote(
                        player,
                        str(msg.get("target", "")),
                        rt_ms=msg.get("rt_ms", 0),
                    )
                    _advance_meeting_bots()
                    await _broadcast_all()
                except ValueError as exc:
                    await websocket.send_json({"error": str(exc)})
            elif mtype == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.pop(websocket, None)
        held_keys[player] = {"w": False, "a": False, "s": False, "d": False}


if __name__ == "__main__":
    lan_ip = get_local_ip()
    print("=" * 64)
    print("  Mini-Among Us 2D Real-Time Server (10 Hz + WebSockets)")
    print(f"  Local URL : http://127.0.0.1:{SERVER_PORT}")
    print(f"  LAN URL   : http://{lan_ip}:{SERVER_PORT}")
    print(f"  Log file  : {get_games_log_path()}")
    print("=" * 64)
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")
