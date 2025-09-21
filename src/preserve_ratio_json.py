import json
import random
import math
from pathlib import Path

# ---------- Settings ----------
INPUT_FILE  = r"C:\Hackathon\crowd_demo_escape_latest.json"
OUTPUT_FILE = r"C:\Hackathon\crowd_demo_escape_lastest_preserve15.json"
SPEED_THRESH = 2.7       # Preserve all phones with any track point speed > 2.7
DELETE_FRACTION = 0.80   # Delete 1/5 of all phones
SEED = 42                # For reproducibility
# -----------------------------

def max_speed_of_phone(phone):
    """
    phone: {"phone_id": str, "track": [ { "t":..., "lon":..., "lat":..., "speed":...}, ... ]}
    Returns the maximum speed found in the track. Missing/None handled as 0.0.
    """
    track = phone.get("track", [])
    max_spd = 0.0
    for pt in track:
        spd = pt.get("speed", 0.0)
        if isinstance(spd, (int, float)):
            if spd > max_spd:
                max_spd = spd
        else:
            # If speed is malformed, ignore it
            continue
    return max_spd

def main():
    random.seed(SEED)

    # 1) Load JSON (list of phones)
    in_path = Path(INPUT_FILE)
    out_path = Path(OUTPUT_FILE)
    if not in_path.exists():
        print(f"Input file not found: {INPUT_FILE}")
        return

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Input JSON must be a list of phone objects.")
        return

    # 2) Partition phones into fast (always keep) vs. non-fast
    fast_phones = []
    nonfast_phones = []
    for phone in data:
        # Require a phone_id; skip entries without it
        pid = phone.get("phone_id")
        if pid is None:
            # If no phone_id, treat as nonfast but we must keep structure consistent
            nonfast_phones.append(phone)
            continue

        if max_speed_of_phone(phone) > SPEED_THRESH:
            fast_phones.append(phone)
        else:
            nonfast_phones.append(phone)

    total_phones = len(data)
    num_fast = len(fast_phones)
    num_nonfast = len(nonfast_phones)

    # 3) Compute how many phones to delete (20% of total), but only from non-fast group
    target_delete = math.floor(total_phones * DELETE_FRACTION)

    if target_delete <= num_nonfast:
        # We can satisfy deletion target by deleting only from non-fast phones
        num_to_delete_from_nonfast = target_delete
        # Randomly choose phones to delete from non-fast set
        indices = list(range(num_nonfast))
        random.shuffle(indices)
        delete_indices = set(indices[:num_to_delete_from_nonfast])

        kept_nonfast = [p for i, p in enumerate(nonfast_phones) if i not in delete_indices]
        kept = fast_phones + kept_nonfast
        feasible = True
    else:
        # Not enough non-fast phones to meet delete target
        # Keep all fast phones, delete ALL non-fast phones
        kept = fast_phones[:]  # copy
        feasible = False

    # 4) Save result
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    # 5) Report
    kept_count = len(kept)
    deleted_count = total_phones - kept_count
    print("Summary")
    print("-------")
    print(f"Total phones:      {total_phones}")
    print(f"Fast phones (> {SPEED_THRESH} m/s): {num_fast}")
    print(f"Non-fast phones:   {num_nonfast}")
    print(f"Target delete (20%): {target_delete}")
    print(f"Actual deleted:    {deleted_count}")
    print(f"Actual kept:       {kept_count}")
    print(f"Output saved to:   {OUTPUT_FILE}")

    if not feasible:
        print("\n[Note]")
        print("Fast phones constitute ≥ 80% of all phones.")
        print("To honor the rule 'preserve all fast phones', we deleted all non-fast phones,")
        print("which is fewer than the requested 20%. Deleting 20% total would require")
        print("removing some fast phones, which the rule forbids.")

if __name__ == "__main__":
    main()
