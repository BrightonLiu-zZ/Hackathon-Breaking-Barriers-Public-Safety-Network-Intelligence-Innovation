"""
Generate data/expanded_gunshot_sim_56day.csv (8 weeks = 56 days).

Day 1: copied from data/expanded_gunshot_sim.csv UNCHANGED (all 10 event bars
       already labeled is_gunshot=1; mask column dropped if present).
Days 2-56: fresh simulation per day, same physics as generate_expanded_dataset.py,
       with two mild-noise additions:
         1. Gunshot crowd size randomised to 85-100 people per event (vs fixed 100).
         2. ~40% of days include a brief 'mini-dispersal' false alarm: 15-25 people
            walk outward for ~12.5 s then resume normal behaviour.  This creates a
            weaker-but-similar signal that the model must learn to distinguish.

Temporal split used by train_xgboost_56day.py:
  train = days 1-39, val = days 40-47, test = days 48-56

Run from repo root: python scripts/generate_56day_dataset.py
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAY1_CSV = ROOT / "data" / "expanded_gunshot_sim.csv"
OUT_CSV = ROOT / "data" / "expanded_gunshot_sim_56day.csv"
OUT_META = ROOT / "data" / "sim_metadata_56day.json"

# ── Simulation constants (match generate_expanded_dataset.py) ────────────────
DT = 2.5
T_DAY = 43200.0
N_DAYS = 56

CENTER_LON, CENTER_LAT = -121.7493613, 38.5411082
AREA_HALF = 25.0
MIN_PEOPLE, MAX_PEOPLE = 50, 150

RUN_MIN, RUN_MAX = 2.0, 2.6
WALK_MIN, WALK_MAX = 1.0, 1.3
STILL_SPEED = 0.0
RATIO_RUN, RATIO_WALK = 0.70, 0.28

NUM_BARS_EVENT = 10
EVENT_DURATION = NUM_BARS_EVENT * DT   # 25.0 s

HEADING_JITTER_NORMAL = math.radians(5)
HEADING_JITTER_AWAY = math.radians(12)

# Per-event gunshot crowd (randomised per day for signal variability)
GUNSHOT_POP_MIN = 85
GUNSHOT_POP_MAX = 100

# Mini-dispersal false-alarm parameters
MINI_DISPERSAL_CHANCE = 0.40       # probability per day
MINI_DISPERSAL_N_MIN = 15          # people affected (min)
MINI_DISPERSAL_N_MAX = 25          # people affected (max)
MINI_DISPERSAL_TICKS = 5           # duration: 5 × 2.5 s = 12.5 s

IMEI_DAY_STRIDE = 10_000_000


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
    lat = math.radians(lat_deg)
    m_lat = 111_132.954 - 559.822 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)
    m_lon = (math.pi / 180) * 6_378_137.0 * math.cos(lat)
    return m_lat, m_lon


_M_PER_DEG_LAT, _M_PER_DEG_LON = _meters_per_degree(CENTER_LAT)


def _to_latlon(dx_m: float, dy_m: float) -> tuple[float, float]:
    return CENTER_LAT + dy_m / _M_PER_DEG_LAT, CENTER_LON + dx_m / _M_PER_DEG_LON


def _luhn_check(base14: str) -> int:
    s = 0
    for i, d in enumerate(reversed(list(map(int, base14)))):
        if i % 2 == 0:
            s += d
        else:
            dbl = d * 2
            s += dbl if dbl < 10 else dbl - 9
    return (10 - (s % 10)) % 10


# ── DaySimulator56 ────────────────────────────────────────────────────────────

class DaySimulator56:
    """
    Self-contained per-day simulation for the 56-day dataset.

    Differences from the 14-day DaySimulator:
    - No mask column; all NUM_BARS_EVENT ticks within the event window are is_gunshot=1.
    - Gunshot crowd size is a parameter (85-100) rather than fixed 100.
    - run() accepts a pre-computed mini-dispersal decision (start tick or None).
    """

    def __init__(self, imei_base: int) -> None:
        self._imei_serial = imei_base
        self.active: dict = {}
        self.retired: set = set()

    # ── IMEI ─────────────────────────────────────────────────────────────────

    def _next_imei(self) -> str:
        base14 = str(self._imei_serial)
        self._imei_serial += 1
        return base14 + str(_luhn_check(base14))

    # ── Spawn / state helpers ─────────────────────────────────────────────────

    def _edge_spawn(self, rng: random.Random) -> tuple[float, float, float]:
        side = rng.choice(["left", "right", "top", "bottom"])
        if side == "left":
            x, y = -AREA_HALF, rng.uniform(-AREA_HALF, AREA_HALF)
        elif side == "right":
            x, y = AREA_HALF, rng.uniform(-AREA_HALF, AREA_HALF)
        elif side == "top":
            x, y = rng.uniform(-AREA_HALF, AREA_HALF), AREA_HALF
        else:
            x, y = rng.uniform(-AREA_HALF, AREA_HALF), -AREA_HALF
        h = math.atan2(-y, -x) + rng.uniform(-HEADING_JITTER_NORMAL, HEADING_JITTER_NORMAL)
        return x, y, h

    def _choose_state(self, rng: random.Random) -> str:
        r = rng.random()
        if r < RATIO_RUN:
            return "run"
        if r < RATIO_RUN + RATIO_WALK:
            return "walk"
        return "still"

    def _speed(self, state: str, rng: random.Random) -> float:
        if state in ("run", "run_away"):
            return rng.uniform(RUN_MIN, RUN_MAX)
        if state in ("walk", "walk_away"):
            return rng.uniform(WALK_MIN, WALK_MAX)
        return STILL_SPEED

    def _spawn(self, t: float, rng: random.Random) -> None:
        pid = self._next_imei()
        x, y, h = self._edge_spawn(rng)
        st = self._choose_state(rng)
        self.active[pid] = {"x": x, "y": y, "heading": h,
                            "speed": self._speed(st, rng), "state": st}

    # ── Population management ─────────────────────────────────────────────────

    def _maintain(self, t: float, lo: int, hi: int, rng: random.Random) -> None:
        n = len(self.active)
        if n < lo:
            for _ in range(lo - n):
                self._spawn(t, rng)
        elif n <= hi and rng.random() < 0.05:
            self._spawn(t, rng)

    def _adjust_states(self, rng: random.Random) -> None:
        for p in self.active.values():
            if p["state"] not in ("dead", "run_away", "walk_away") and rng.random() < 0.02:
                st = self._choose_state(rng)
                p["state"] = st
                p["speed"] = self._speed(st, rng)

    def _force_pop(self, pop_size: int, t: float, rng: random.Random) -> None:
        """Snap active population to exactly pop_size."""
        n = len(self.active)
        if n < pop_size:
            for _ in range(pop_size - n):
                self._spawn(t, rng)
        elif n > pop_size:
            for pid in rng.sample(list(self.active.keys()), n - pop_size):
                self.retired.add(pid)
                del self.active[pid]

    def _snap_kill(self, k: int, radius: float, rng: random.Random) -> None:
        alive = [pid for pid, p in self.active.items() if p["state"] != "dead"]
        for pid in rng.sample(alive, min(k, len(alive))):
            a = rng.uniform(0, 2 * math.pi)
            self.active[pid].update(
                {"x": radius * math.cos(a), "y": radius * math.sin(a),
                 "state": "dead", "speed": STILL_SPEED}
            )

    # ── Gunshot phase transitions ─────────────────────────────────────────────

    def _phase1(self, rng: random.Random) -> None:
        """At event start: mixed walk/still/run crowd."""
        for p in self.active.values():
            if p["state"] != "dead":
                p["state"] = "walk"
                p["speed"] = self._speed("walk", rng)
        alive = [pid for pid, p in self.active.items() if p["state"] != "dead"]
        for pid in rng.sample(alive, min(25, len(alive))):
            self.active[pid]["state"] = "still"
            self.active[pid]["speed"] = STILL_SPEED
        still_set = {pid for pid in alive if self.active[pid]["state"] == "still"}
        remaining = [pid for pid in alive if pid not in still_set]
        for pid in rng.sample(remaining, min(2, len(remaining))):
            self.active[pid]["state"] = "run"
            self.active[pid]["speed"] = self._speed("run", rng)
        self._snap_kill(3, 2.0, rng)

    def _phase2(self, rng: random.Random) -> None:
        """At +2.5 s: scatter majority run_away."""
        self._snap_kill(2, 5.0, rng)
        alive = [pid for pid, p in self.active.items() if p["state"] != "dead"]
        rng.shuffle(alive)
        n = len(alive)
        for pid in alive[: max(0, n - 10)]:
            self.active[pid]["state"] = "run_away"
            self.active[pid]["speed"] = self._speed("run_away", rng)
        for pid in alive[max(0, n - 10): max(0, n - 5)]:
            self.active[pid]["state"] = "walk_away"
            self.active[pid]["speed"] = self._speed("walk_away", rng)
        for pid in alive[max(0, n - 5):]:
            self.active[pid]["state"] = "still"
            self.active[pid]["speed"] = STILL_SPEED

    def _phase3(self, rng: random.Random) -> None:
        """At +5.0 s: all run away."""
        for p in self.active.values():
            if p["state"] != "dead":
                p["state"] = "run_away"
                p["speed"] = self._speed("run_away", rng)

    # ── Step ──────────────────────────────────────────────────────────────────

    def _step(self, person: dict, in_gunshot: bool, rng: random.Random) -> None:
        st = person["state"]
        x, y, h, v = person["x"], person["y"], person["heading"], person["speed"]
        if st in ("still", "dead"):
            pass
        elif st in ("run", "walk"):
            h += rng.uniform(-HEADING_JITTER_NORMAL, HEADING_JITTER_NORMAL)
            x += v * math.cos(h) * DT
            y += v * math.sin(h) * DT
        elif st in ("run_away", "walk_away"):
            h = math.atan2(y, x) + rng.uniform(-HEADING_JITTER_AWAY, HEADING_JITTER_AWAY)
            x += v * math.cos(h) * DT
            y += v * math.sin(h) * DT
        if in_gunshot:
            x = max(-AREA_HALF, min(AREA_HALF, x))
            y = max(-AREA_HALF, min(AREA_HALF, y))
        person["x"], person["y"], person["heading"] = x, y, h

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(
        self,
        event_start: float,
        t_offset: float,
        rng: random.Random,
        gunshot_pop: int,
        mini_start: float | None,
    ) -> pd.DataFrame:
        """
        Simulate one day.
        event_start: local-day time (multiple of DT) when the gunshot event begins.
        t_offset: added to all output t values (global time axis).
        gunshot_pop: number of people forced present at event start.
        mini_start: local-day tick for mini-dispersal false alarm, or None.
        """
        self.active = {}
        self.retired = set()

        event_end = event_start + EVENT_DURATION
        event_ticks = {round(event_start + i * DT, 4) for i in range(NUM_BARS_EVENT)}

        num_steps = int(T_DAY / DT) + 1
        times = [round(i * DT, 4) for i in range(num_steps)]

        mini_end: float | None = (
            round(mini_start + MINI_DISPERSAL_TICKS * DT, 4) if mini_start is not None else None
        )

        mini_pids: set = set()
        phase1_done = phase2_done = phase3_done = False
        rows: list[dict] = []

        for t in times:
            t_key = round(t, 4)
            in_gunshot = event_start <= t < event_end

            # ── Normal crowd maintenance ──────────────────────────────────────
            if not in_gunshot:
                lo, hi = MIN_PEOPLE, MAX_PEOPLE
                if event_start - 60 <= t < event_start - 10:
                    lo, hi = 90, 110
                elif event_start - 10 <= t < event_start:
                    lo, hi = 98, 102
                self._maintain(t, lo, hi, rng)
                self._adjust_states(rng)

            # ── Gunshot phase transitions ─────────────────────────────────────
            if in_gunshot:
                if not phase1_done and abs(t - event_start) < 1e-3:
                    self._force_pop(gunshot_pop, t, rng)
                    self._phase1(rng)
                    phase1_done = True
                elif phase1_done and not phase2_done and abs(t - (event_start + DT)) < 1e-3:
                    self._force_pop(gunshot_pop, t, rng)
                    self._phase2(rng)
                    phase2_done = True
                elif phase2_done and not phase3_done and abs(t - (event_start + 2 * DT)) < 1e-3:
                    self._force_pop(gunshot_pop, t, rng)
                    self._phase3(rng)
                    phase3_done = True

            # ── Mini-dispersal trigger ────────────────────────────────────────
            if mini_start is not None and abs(t - mini_start) < 1e-3:
                candidates = [
                    pid for pid, p in self.active.items()
                    if p["state"] not in ("dead",)
                    and abs(p["x"]) < AREA_HALF * 0.8
                    and abs(p["y"]) < AREA_HALF * 0.8
                ]
                n_scatter = rng.randint(MINI_DISPERSAL_N_MIN, MINI_DISPERSAL_N_MAX)
                mini_pids = set(rng.sample(candidates, min(n_scatter, len(candidates))))
                for pid in mini_pids:
                    self.active[pid]["state"] = "walk_away"
                    self.active[pid]["speed"] = self._speed("walk_away", rng)

            # ── Mini-dispersal recovery ───────────────────────────────────────
            if mini_end is not None and abs(t - mini_end) < 1e-3 and mini_pids:
                for pid in mini_pids:
                    if pid in self.active:
                        st = rng.choice(["walk", "still"])
                        self.active[pid]["state"] = st
                        self.active[pid]["speed"] = self._speed(st, rng)
                mini_pids = set()

            # ── Step all active people ────────────────────────────────────────
            to_remove = []
            for pid, person in self.active.items():
                self._step(person, in_gunshot, rng)
                if not in_gunshot and (abs(person["x"]) > AREA_HALF or abs(person["y"]) > AREA_HALF):
                    to_remove.append(pid)
            for pid in to_remove:
                self.retired.add(pid)
                del self.active[pid]

            if in_gunshot and len(self.active) != gunshot_pop:
                self._force_pop(gunshot_pop, t, rng)

            # ── Label: all event-window ticks are positive ────────────────────
            is_gs = 1 if t_key in event_ticks else 0

            t_global = round(t + t_offset, 1)
            for pid, p in self.active.items():
                lat, lon = _to_latlon(p["x"], p["y"])
                rows.append({
                    "phone_id": pid,
                    "t": t_global,
                    "lat": round(lat, 8),
                    "lon": round(lon, 8),
                    "is_gunshot": is_gs,
                })

        return pd.DataFrame(rows, columns=["phone_id", "t", "lat", "lon", "is_gunshot"])


# ── Day 1 loader ──────────────────────────────────────────────────────────────

def _load_day1() -> pd.DataFrame:
    """
    Read original Day 1 CSV unchanged. mask column dropped if present (not used here).
    The original already has all 10 event bars labeled is_gunshot=1.
    """
    df = pd.read_csv(DAY1_CSV)
    df = df.drop(columns=["mask"], errors="ignore")
    return df[["phone_id", "t", "lat", "lon", "is_gunshot"]]


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Day 1: reading {DAY1_CSV} ...")
    day1_df = _load_day1()
    n_pos_d1 = int(day1_df["is_gunshot"].sum())
    print(f"  Rows: {len(day1_df):,}  is_gunshot=1: {n_pos_d1}")

    day1_event_ticks = sorted(day1_df.loc[day1_df["is_gunshot"] == 1, "t"].unique())
    event_starts_global: list[float] = [
        float(day1_event_ticks[0]) if day1_event_ticks else float("nan")
    ]

    all_dfs: list[pd.DataFrame] = [day1_df]

    for day_idx in range(2, N_DAYS + 1):
        t_offset = (day_idx - 1) * T_DAY

        # Event placement seed (separate from simulation RNG)
        event_rng = random.Random(day_idx * 1000)
        raw_start = event_rng.uniform(7200.0, T_DAY - 7200.0)
        event_start = round(raw_start / DT) * DT

        # Gunshot pop uses its own seed so it doesn't perturb sim_rng
        pop_rng = random.Random(day_idx * 7919)
        gunshot_pop = pop_rng.randint(GUNSHOT_POP_MIN, GUNSHOT_POP_MAX)

        # Main simulation RNG
        sim_rng = random.Random(day_idx)

        # Mini-dispersal: decide now (uses sim_rng, so reproducible)
        mini_start: float | None = None
        if sim_rng.random() < MINI_DISPERSAL_CHANCE:
            event_end = event_start + EVENT_DURATION
            # Candidate ticks at least 120 s away from the real event
            num_steps = int(T_DAY / DT) + 1
            times_all = [round(i * DT, 4) for i in range(num_steps)]
            far = [ti for ti in times_all
                   if ti < event_start - 120 or ti > event_end + 120]
            if far:
                mid = far[len(far) // 4: 3 * len(far) // 4]
                mini_start = sim_rng.choice(mid if mid else far)

        global_event_t = event_start + t_offset
        mini_info = f"  mini@{mini_start:.0f}s" if mini_start is not None else ""
        print(
            f"Day {day_idx:2d}: event={event_start:.1f}s  global={global_event_t:.1f}s"
            f"  pop={gunshot_pop}{mini_info}"
        )

        imei_base = 10_000_000_000_000 + day_idx * IMEI_DAY_STRIDE
        sim = DaySimulator56(imei_base)
        day_df = sim.run(event_start, t_offset, sim_rng, gunshot_pop, mini_start)
        all_dfs.append(day_df)
        event_starts_global.append(float(global_event_t))

    result = pd.concat(all_dfs, ignore_index=True)
    result.to_csv(OUT_CSV, index=False)

    n_pos = int(result["is_gunshot"].sum())
    print(f"\nWrote {OUT_CSV}")
    print(f"  Total rows: {len(result):,}  is_gunshot=1: {n_pos}")
    print(f"  Positive rate: {n_pos / len(result):.6f}")

    meta = {
        "n_days": N_DAYS,
        "day_duration_s": T_DAY,
        "dt_s": DT,
        "num_bars_event": NUM_BARS_EVENT,
        "event_duration_s": EVENT_DURATION,
        "positive_bars_per_event": NUM_BARS_EVENT,
        "gunshot_pop_range": [GUNSHOT_POP_MIN, GUNSHOT_POP_MAX],
        "mini_dispersal_chance": MINI_DISPERSAL_CHANCE,
        "mini_dispersal_n_range": [MINI_DISPERSAL_N_MIN, MINI_DISPERSAL_N_MAX],
        "mini_dispersal_ticks": MINI_DISPERSAL_TICKS,
        "event_starts_global_s": event_starts_global,
        "temporal_split": {
            "train_days": "1-39",
            "val_days": "40-47",
            "test_days": "48-56",
        },
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_META}")


if __name__ == "__main__":
    main()
