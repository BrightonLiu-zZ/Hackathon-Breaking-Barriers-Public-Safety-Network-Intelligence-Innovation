"""
Generate data/expanded_gunshot_sim_14day.csv:

Day 1 (t in [0, 43200)): copied from data/expanded_gunshot_sim.csv (UNCHANGED).
  - mask=0 for all rows except bars 2-10 of the gunshot event, which get mask=1
  - is_gunshot column unchanged from original CSV

Days 2-14 (t offset by (day-1)*43200):
  - Fresh simulation per day (same physics as generate_expanded_dataset.py)
  - Event time: random in [7200, 35999], snapped to nearest 2.5s tick, seed=day_index
  - Bar 1 of event: is_gunshot=1, mask=0
  - Bars 2-10 of event: is_gunshot=0, mask=1
  - All other rows: is_gunshot=0, mask=0

Run from repo root: python scripts/generate_14day_dataset.py
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAY1_CSV = ROOT / "data" / "expanded_gunshot_sim.csv"
OUT_CSV = ROOT / "data" / "expanded_gunshot_sim_14day.csv"
OUT_META = ROOT / "data" / "sim_metadata_14day.json"

# ── Simulation constants (match generate_expanded_dataset.py) ───────────────
DT = 2.5
T_DAY = 43200.0
N_DAYS = 14

CENTER_LON, CENTER_LAT = -121.7493613, 38.5411082
AREA_HALF = 25.0
MIN_PEOPLE, MAX_PEOPLE = 50, 150

RUN_MIN, RUN_MAX = 2.0, 2.6
WALK_MIN, WALK_MAX = 1.0, 1.3
STILL_SPEED = 0.0

RATIO_RUN, RATIO_WALK = 0.70, 0.28

NUM_BARS_EVENT = 10          # bars spanned by the event window (10 × 2.5s = 25s)
EVENT_DURATION = NUM_BARS_EVENT * DT   # 25.0 s

GUNSHOT_POP = 100
HEADING_JITTER_NORMAL = math.radians(5)
HEADING_JITTER_AWAY = math.radians(12)

# Spacing between per-day IMEI serial pools (must exceed max phones per day)
IMEI_DAY_STRIDE = 10_000_000


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


# ── DaySimulator ─────────────────────────────────────────────────────────────

class DaySimulator:
    """Self-contained per-day simulation state — no module-level globals."""

    def __init__(self, imei_base: int):
        self._imei_serial = imei_base
        self.active: dict = {}
        self.retired: set = set()

    # ── IMEI ─────────────────────────────────────────────────────────────────

    def _next_imei(self) -> str:
        base14 = str(self._imei_serial)
        self._imei_serial += 1
        return base14 + str(_luhn_check(base14))

    # ── Spawn / state ─────────────────────────────────────────────────────────

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

    def _force_100(self, t: float, rng: random.Random) -> None:
        n = len(self.active)
        if n < GUNSHOT_POP:
            for _ in range(GUNSHOT_POP - n):
                self._spawn(t, rng)
        elif n > GUNSHOT_POP:
            for pid in rng.sample(list(self.active.keys()), n - GUNSHOT_POP):
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

    # ── Phase transitions ─────────────────────────────────────────────────────

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
        """At +2.5s: scatter majority run_away."""
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
        """At +5.0s: all run away."""
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

    def run(self, event_start: float, t_offset: float, rng: random.Random) -> pd.DataFrame:
        """
        Simulate one day. event_start must be a multiple of DT.
        t_offset is added to all output t values (global time axis).
        Returns DataFrame with columns: phone_id, t, lat, lon, is_gunshot, mask
        """
        self.active = {}
        self.retired = set()

        event_end = event_start + EVENT_DURATION
        # Exact tick set for the event window
        event_ticks = {round(event_start + i * DT, 4) for i in range(NUM_BARS_EVENT)}
        first_tick = round(event_start, 4)

        num_steps = int(T_DAY / DT) + 1
        times = [round(i * DT, 4) for i in range(num_steps)]

        phase1_done = phase2_done = phase3_done = False
        rows: list[dict] = []

        for t in times:
            in_gunshot = (event_start <= t < event_end)

            if not in_gunshot:
                lo, hi = MIN_PEOPLE, MAX_PEOPLE
                if event_start - 60 <= t < event_start - 10:
                    lo, hi = 90, 110
                elif event_start - 10 <= t < event_start:
                    lo, hi = 98, 102
                self._maintain(t, lo, hi, rng)
                self._adjust_states(rng)

            if in_gunshot:
                if not phase1_done and abs(t - event_start) < 1e-3:
                    self._force_100(t, rng)
                    self._phase1(rng)
                    phase1_done = True
                elif phase1_done and not phase2_done and abs(t - (event_start + DT)) < 1e-3:
                    self._force_100(t, rng)
                    self._phase2(rng)
                    phase2_done = True
                elif phase2_done and not phase3_done and abs(t - (event_start + 2 * DT)) < 1e-3:
                    self._force_100(t, rng)
                    self._phase3(rng)
                    phase3_done = True

            to_remove = []
            for pid, person in self.active.items():
                self._step(person, in_gunshot, rng)
                if not in_gunshot and (abs(person["x"]) > AREA_HALF or abs(person["y"]) > AREA_HALF):
                    to_remove.append(pid)
            for pid in to_remove:
                self.retired.add(pid)
                del self.active[pid]

            if in_gunshot and len(self.active) != GUNSHOT_POP:
                self._force_100(t, rng)

            # Determine label + mask for this tick
            t_key = round(t, 4)
            if t_key in event_ticks:
                is_gs = 1 if t_key == first_tick else 0
                mask = 0 if t_key == first_tick else 1
            else:
                is_gs = 0
                mask = 0

            t_global = round(t + t_offset, 1)
            for pid, p in self.active.items():
                lat, lon = _to_latlon(p["x"], p["y"])
                rows.append({
                    "phone_id": pid,
                    "t": t_global,
                    "lat": round(lat, 8),
                    "lon": round(lon, 8),
                    "is_gunshot": is_gs,
                    "mask": mask,
                })

        return pd.DataFrame(rows, columns=["phone_id", "t", "lat", "lon", "is_gunshot", "mask"])


# ── Day 1 helper ─────────────────────────────────────────────────────────────

def _process_day1() -> pd.DataFrame:
    """
    Read existing day-1 CSV (never modified), add mask column.
    Bar 1 of the event: mask=0. Bars 2-10: mask=1.
    """
    df = pd.read_csv(DAY1_CSV)
    df["mask"] = 0

    event_ticks = sorted(df.loc[df["is_gunshot"] == 1, "t"].unique())
    if event_ticks:
        masked_ticks = set(event_ticks[1:])   # bars 2-10
        df.loc[(df["is_gunshot"] == 1) & (df["t"].isin(masked_ticks)), "mask"] = 1

    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Day 1: reading {DAY1_CSV} ...")
    day1_df = _process_day1()

    day1_event_ticks = sorted(day1_df.loc[day1_df["is_gunshot"] == 1, "t"].unique())
    print(f"  Day 1 event ticks (is_gunshot=1): {day1_event_ticks}")
    print(f"  Day 1 masked bars: {int((day1_df['mask'] == 1).sum() // day1_df.loc[day1_df['mask']==1,'phone_id'].nunique() if day1_df['mask'].sum() > 0 else 0)}")

    all_dfs = [day1_df]
    event_starts_global: list[float] = [float(day1_event_ticks[0]) if day1_event_ticks else float("nan")]

    for day_idx in range(2, N_DAYS + 1):
        t_offset = (day_idx - 1) * T_DAY

        # Sample event start for this day (reproducible per day)
        day_rng_for_event = random.Random(day_idx * 1000)   # separate seed for event placement
        raw_start = day_rng_for_event.uniform(7200.0, T_DAY - 7200.0)
        event_start = round(raw_start / DT) * DT             # snap to DT grid

        print(f"Day {day_idx}: event_start={event_start:.1f}s  global_t={event_start + t_offset:.1f}s")

        # Day simulation uses day_idx as seed (distinct from event-placement seed)
        sim_rng = random.Random(day_idx)
        imei_base = 10_000_000_000_000 + day_idx * IMEI_DAY_STRIDE
        sim = DaySimulator(imei_base)
        day_df = sim.run(event_start, t_offset, sim_rng)
        all_dfs.append(day_df)

        event_starts_global.append(float(event_start + t_offset))

    result = pd.concat(all_dfs, ignore_index=True)
    result.to_csv(OUT_CSV, index=False)
    n_pos = int(result["is_gunshot"].sum())
    n_mask = int(result["mask"].sum())
    print(f"\nWrote {OUT_CSV}")
    print(f"  Rows: {len(result):,}  is_gunshot=1: {n_pos}  masked: {n_mask}")

    meta = {
        "n_days": N_DAYS,
        "day_duration_s": T_DAY,
        "dt_s": DT,
        "num_bars_event": NUM_BARS_EVENT,
        "event_duration_s": EVENT_DURATION,
        "positive_bars_per_event": 1,
        "masked_bars_per_event": NUM_BARS_EVENT - 1,
        "event_starts_global_s": event_starts_global,
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_META}")


if __name__ == "__main__":
    main()
