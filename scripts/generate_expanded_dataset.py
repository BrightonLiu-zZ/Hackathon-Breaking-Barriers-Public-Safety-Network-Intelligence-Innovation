"""
Regenerate data/expanded_gunshot_sim.csv without Jupyter (same logic as notebooks/create_expanded_dataset.ipynb).
Run from repo root: python scripts/generate_expanded_dataset.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "expanded_gunshot_sim.csv"

random.seed(42)

DT = 2.5
T_START = 0.0
T_END = 43200.0
CENTER_LON, CENTER_LAT = -121.7493613, 38.5411082
AREA_HALF = 25.0
MIN_PEOPLE, MAX_PEOPLE = 50, 150

RUN_MIN, RUN_MAX = 2.0, 2.6
WALK_MIN, WALK_MAX = 1.0, 1.3
STILL_SPEED = 0.0

RATIO_RUN, RATIO_WALK, RATIO_STILL = 0.70, 0.28, 0.02

NUM_GUNSHOT_EVENTS = 1
EVENT_DURATION = 25.0
MIN_EVENT_GAP_S = 7200.0

GUNSHOT_POP = 100
HEADING_JITTER_NORMAL = math.radians(5)
HEADING_JITTER_AWAY = math.radians(12)


def sample_event_starts(num_events: int, t_end: float, gap_s: float, seed: int = 42) -> list[float]:
    rnd = random.Random(seed)
    starts: list[float] = []
    for _ in range(num_events):
        for _attempt in range(5000):
            s = rnd.uniform(7200.0, t_end - 7200.0)
            if all(abs(s - x) >= gap_s for x in starts):
                starts.append(s)
                break
        else:
            raise RuntimeError("Could not place gunshot events; increase T_END or reduce count")
    return sorted(starts)


GUNSHOT_EVENT_STARTS = sample_event_starts(NUM_GUNSHOT_EVENTS, T_END, MIN_EVENT_GAP_S)


def meters_per_degree(lat_deg: float) -> tuple[float, float]:
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111_132.954 - 559.822 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)
    m_per_deg_lon = (math.pi / 180) * 6_378_137.0 * math.cos(lat)
    return m_per_deg_lat, m_per_deg_lon


M_PER_DEG_LAT, M_PER_DEG_LON = meters_per_degree(CENTER_LAT)


def meters_to_latlon(dx_m: float, dy_m: float, center_lat: float, center_lon: float) -> tuple[float, float]:
    dlat = dy_m / M_PER_DEG_LAT
    dlon = dx_m / M_PER_DEG_LON
    return center_lat + dlat, center_lon + dlon


def luhn_checksum14(num_str14: str) -> int:
    s = 0
    reverse_digits = list(map(int, num_str14[::-1]))
    for i, d in enumerate(reverse_digits):
        if i % 2 == 0:
            s += d
        else:
            dbl = d * 2
            s += dbl if dbl < 10 else dbl - 9
    return (10 - (s % 10)) % 10


next_imei_serial = 10000000000000


def generate_imei() -> str:
    global next_imei_serial
    base14 = str(next_imei_serial)
    next_imei_serial += 1
    cksum = luhn_checksum14(base14)
    return base14 + str(cksum)


def random_edge_spawn() -> tuple[float, float, float]:
    side = random.choice(["left", "right", "top", "bottom"])
    if side == "left":
        x = -AREA_HALF
        y = random.uniform(-AREA_HALF, AREA_HALF)
        heading_center = math.atan2(0 - y, 0 - x)
    elif side == "right":
        x = AREA_HALF
        y = random.uniform(-AREA_HALF, AREA_HALF)
        heading_center = math.atan2(0 - y, 0 - x)
    elif side == "top":
        y = AREA_HALF
        x = random.uniform(-AREA_HALF, AREA_HALF)
        heading_center = math.atan2(0 - y, 0 - x)
    else:
        y = -AREA_HALF
        x = random.uniform(-AREA_HALF, AREA_HALF)
        heading_center = math.atan2(0 - y, 0 - x)
    heading = heading_center + random.uniform(-HEADING_JITTER_NORMAL, HEADING_JITTER_NORMAL)
    return x, y, heading


def choose_state_for_normal() -> str:
    r = random.random()
    if r < RATIO_RUN:
        return "run"
    if r < RATIO_RUN + RATIO_WALK:
        return "walk"
    return "still"


def speed_for_state(state: str) -> float:
    if state in ("run", "run_away"):
        return random.uniform(RUN_MIN, RUN_MAX)
    if state in ("walk", "walk_away"):
        return random.uniform(WALK_MIN, WALK_MAX)
    return STILL_SPEED


def clamp_to_area(x: float, y: float) -> tuple[float, float]:
    return max(-AREA_HALF, min(AREA_HALF, x)), max(-AREA_HALF, min(AREA_HALF, y))


def step_person(person: dict, dt: float, t_now: float, gunshot_window: bool = False) -> None:
    st = person["state"]
    x, y, h, v = person["x"], person["y"], person["heading"], person["speed"]

    if st in ("still", "dead"):
        pass
    elif st in ("run", "walk"):
        h += random.uniform(-HEADING_JITTER_NORMAL, HEADING_JITTER_NORMAL)
        x += v * math.cos(h) * dt
        y += v * math.sin(h) * dt
    elif st in ("run_away", "walk_away"):
        away_angle = math.atan2(y, x)
        h = away_angle + random.uniform(-HEADING_JITTER_AWAY, HEADING_JITTER_AWAY)
        x += v * math.cos(h) * dt
        y += v * math.sin(h) * dt

    if gunshot_window:
        x, y = clamp_to_area(x, y)

    person["x"], person["y"], person["heading"] = x, y, h


active: dict = {}
retired_ids: set = set()


def spawn_one(now_t: float) -> None:
    pid = generate_imei()
    x, y, heading = random_edge_spawn()
    st = choose_state_for_normal()
    v = speed_for_state(st)
    active[pid] = {
        "id": pid,
        "x": x,
        "y": y,
        "heading": heading,
        "speed": v,
        "state": st,
        "born_t": now_t,
        "exited": False,
    }


def try_exit(person: dict) -> bool:
    return (
        person["x"] < -AREA_HALF
        or person["x"] > AREA_HALF
        or person["y"] < -AREA_HALF
        or person["y"] > AREA_HALF
    )


def maintain_normal_population(now_t: float, target_min: int = MIN_PEOPLE, target_max: int = MAX_PEOPLE) -> None:
    n = len(active)
    if n < target_min:
        for _ in range(target_min - n):
            spawn_one(now_t)
    elif n > target_max:
        pass
    else:
        if random.random() < 0.05:
            spawn_one(now_t)


def adjust_states_toward_ratio() -> None:
    for p in active.values():
        if p["state"] in ("dead", "run_away", "walk_away"):
            continue
        if random.random() < 0.02:
            st = choose_state_for_normal()
            p["state"] = st
            p["speed"] = speed_for_state(st)


def force_population_to_exact_100(now_t: float) -> None:
    n = len(active)
    if n < GUNSHOT_POP:
        for _ in range(GUNSHOT_POP - n):
            spawn_one(now_t)
    elif n > GUNSHOT_POP:
        extras = random.sample(list(active.keys()), n - GUNSHOT_POP)
        for pid in extras:
            retired_ids.add(pid)
            del active[pid]


def snap_random_to_ring_and_kill(k: int, radius_m: float) -> list:
    alive_pids = [pid for pid, p in active.items() if p["state"] != "dead"]
    chosen = random.sample(alive_pids, k)
    for pid in chosen:
        angle = random.uniform(0, 2 * math.pi)
        active[pid]["x"] = radius_m * math.cos(angle)
        active[pid]["y"] = radius_m * math.sin(angle)
        active[pid]["state"] = "dead"
        active[pid]["speed"] = STILL_SPEED
    return chosen


def set_mix_at_25202p5() -> None:
    for p in active.values():
        if p["state"] != "dead":
            p["state"] = "walk"
            p["speed"] = speed_for_state("walk")
    alive_pids = [pid for pid, p in active.items() if p["state"] != "dead"]
    still_ids = random.sample(alive_pids, 25)
    for pid in still_ids:
        active[pid]["state"] = "still"
        active[pid]["speed"] = STILL_SPEED
    remaining = [pid for pid in alive_pids if pid not in still_ids]
    run_ids = random.sample(remaining, 2)
    for pid in run_ids:
        active[pid]["state"] = "run"
        active[pid]["speed"] = speed_for_state("run")
    snap_random_to_ring_and_kill(3, 2.0)


def set_mix_at_25205() -> None:
    snap_random_to_ring_and_kill(2, 5.0)
    alive_pids = [pid for pid, p in active.items() if p["state"] != "dead"]
    assert len(alive_pids) == 95
    random.shuffle(alive_pids)
    runA = alive_pids[:85]
    walkA = alive_pids[85:90]
    stillA = alive_pids[90:]
    for pid in runA:
        active[pid]["state"] = "run_away"
        active[pid]["speed"] = speed_for_state("run_away")
    for pid in walkA:
        active[pid]["state"] = "walk_away"
        active[pid]["speed"] = speed_for_state("walk_away")
    for pid in stillA:
        active[pid]["state"] = "still"
        active[pid]["speed"] = STILL_SPEED


def set_all_alive_run_away_at_25207p5() -> None:
    for p in active.values():
        if p["state"] != "dead":
            p["state"] = "run_away"
            p["speed"] = speed_for_state("run_away")


def gunshot_window_active(t: float) -> bool:
    for s in GUNSHOT_EVENT_STARTS:
        if s <= t <= s + EVENT_DURATION:
            return True
    return False


def active_event_start(t: float) -> float | None:
    for s in GUNSHOT_EVENT_STARTS:
        if s <= t <= s + EVENT_DURATION:
            return s
    return None


def target_bounds_for_time(t: float) -> tuple[int, int]:
    if active_event_start(t) is not None:
        return GUNSHOT_POP, GUNSHOT_POP
    for s in sorted(GUNSHOT_EVENT_STARTS):
        if s - 60 <= t < s - 10:
            return 90, 110
        if s - 10 <= t < s:
            return 98, 102
    return MIN_PEOPLE, MAX_PEOPLE


def near_tick(t: float, target: float, eps: float = 1e-6) -> bool:
    return abs(t - target) < eps


def run_simulation() -> pd.DataFrame:
    rows: list[dict] = []

    def record_positions(now_t: float, is_gunshot_flag: bool) -> None:
        for p in active.values():
            lat, lon = meters_to_latlon(p["x"], p["y"], CENTER_LAT, CENTER_LON)
            rows.append(
                {
                    "phone_id": p["id"],
                    "t": round(now_t, 1),
                    "lat": round(lat, 8),
                    "lon": round(lon, 8),
                    "is_gunshot": 1 if is_gunshot_flag else 0,
                }
            )

    num_steps = int((T_END - T_START) / DT) + 1
    times = [T_START + i * DT for i in range(num_steps)]

    print("Gunshot event start times (s):", [round(x, 1) for x in GUNSHOT_EVENT_STARTS])

    for t in times:
        in_gunshot = gunshot_window_active(t)

        if not in_gunshot:
            lo, hi = target_bounds_for_time(t)
            maintain_normal_population(t, lo, hi)
            adjust_states_toward_ratio()

        for s in GUNSHOT_EVENT_STARTS:
            if near_tick(t, s):
                force_population_to_exact_100(t)
                set_mix_at_25202p5()
            if near_tick(t, s + 2.5):
                force_population_to_exact_100(t)
                set_mix_at_25205()
            if near_tick(t, s + 5.0):
                force_population_to_exact_100(t)
                set_all_alive_run_away_at_25207p5()

        to_remove: list = []
        for pid, person in list(active.items()):
            step_person(person, DT, t, gunshot_window=in_gunshot)
            if not in_gunshot and try_exit(person):
                to_remove.append(pid)
        for pid in to_remove:
            retired_ids.add(pid)
            del active[pid]

        if in_gunshot and len(active) != GUNSHOT_POP:
            force_population_to_exact_100(t)

        record_positions(t, is_gunshot_flag=in_gunshot)

    df = pd.DataFrame(rows, columns=["phone_id", "t", "lat", "lon", "is_gunshot"])
    return df


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = run_simulation()
    df.to_csv(OUT_CSV, index=False)
    meta_path = ROOT / "data" / "sim_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "gunshot_event_starts_s": GUNSHOT_EVENT_STARTS,
                "event_duration_s": EVENT_DURATION,
                "t_end_s": T_END,
                "dt_s": DT,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Done. Wrote", OUT_CSV, "with", len(df), "rows and columns:", list(df.columns))
    print("Wrote", meta_path)


if __name__ == "__main__":
    main()
