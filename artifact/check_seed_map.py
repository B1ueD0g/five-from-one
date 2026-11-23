"""Check the registered source-to-seed bijection and a routing-preserving negative control."""
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re


def check_seed_map(rows):
    """Check both directions per question. Different decoding profiles may replay the same tapes."""
    by_item = defaultdict(list)
    for row in rows:
        by_item[row["item_id"]].append(row)
    counts = {}
    for item, family in by_item.items():
        by_source = {}
        by_seed = {}
        for row in family:
            if len(row["seeds"]) != 5 or len(row["source_batches"]) != 5:
                raise ValueError(f"{item}: expected five tape slots")
            for slot, (source_batch, raw) in enumerate(zip(row["source_batches"], row["seeds"], strict=True)):
                if not isinstance(raw, str) or not re.fullmatch(r"[0-9a-fA-F]{16}", raw):
                    raise ValueError(f"{item}: malformed u64 seed")
                seed = int(raw, 16)
                source = slot, source_batch
                if by_source.setdefault(source, seed) != seed:
                    raise ValueError(f"{item}: one source maps to two seeds")
                if by_seed.setdefault(seed, source) != source:
                    raise ValueError(f"{item}: one seed maps to two sources")
        counts[item] = len(by_source)
    return counts


def main(directory):
    schedules = json.loads((directory / "schedule.json").read_bytes())["runs"]
    for name in ("serial", "parallel"):
        rows = schedules[name]
        counts = check_seed_map(rows)
        expected_questions = {"serial": 12, "parallel": 3}[name]
        if len(counts) != expected_questions or any(n != 25 for n in counts.values()):
            raise ValueError(f"{name}: unexpected question or logical-source count")
        print(f"{name}: {len(counts)} questions, {sum(counts.values())} within-question sources, "
              f"{len(rows) * 5} routed cells")
    # Repeated tuples, including reruns on the same question in another profile, keep their identities.
    all_rows = [row for rows in schedules.values() for row in rows]
    if check_seed_map(all_rows) != check_seed_map(schedules["serial"]):
        raise ValueError("repeated profiles failed to preserve the original source-to-seed map")
    print(f"All four schedules: {len(all_rows) * 5} routed cells, same 300 within-question sources.")

    # Keep routing labels and the same-slot match pattern, change only one cross-slot physical identity.
    altered = deepcopy(schedules["serial"])
    item = altered[0]["item_id"]
    source = altered[0]["source_batches"][1]
    replacement = altered[0]["seeds"][0]
    for row in altered:
        if row["item_id"] == item and row["source_batches"][1] == source:
            row["seeds"][1] = replacement
    target_original = [r for r in schedules["serial"] if r["item_id"] == item]
    target_altered = [r for r in altered if r["item_id"] == item]
    for j in range(5):
        original = [r["seeds"][j] for r in target_original]
        recoded = [r["seeds"][j] for r in target_altered]
        if any((a == b) != (c == d) for a, c in zip(original, recoded, strict=True)
               for b, d in zip(original, recoded, strict=True)):
            raise ValueError("negative control changed the same-slot matches")
    try:
        check_seed_map(altered)
    except ValueError as error:
        if "one seed maps to two sources" not in str(error):
            raise
        print("Cross-slot alias rejected with source labels and same-slot matching intact.")
    else:
        raise ValueError("cross-slot seed alias was accepted")


if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
