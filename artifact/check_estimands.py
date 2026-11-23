"""Check an exact majority counterexample and two recorded prefix projections."""
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
from itertools import product
import json

from verify import HERE, parsed, read_runs, reported_rule, require


def xor_majority():
    actual_law, reference_law, directions = Counter(), Counter(), Counter()
    for bits in product((0, 1), repeat=9):
        root = bits[0]
        own, donor_roots = bits[1:5], bits[5:9]
        actual = (root, *(x ^ root for x in own))
        reference = (root, *(x ^ v for x, v in zip(own, donor_roots, strict=True)))
        actual_law[actual] += 1
        reference_law[reference] += 1
        directions[sum(actual) >= 3, sum(reference) >= 3] += 1

    require(len(actual_law) == 32 and set(actual_law.values()) == {16}
            and actual_law == reference_law, "five-bit laws must both be uniform")
    require(directions == {(False, False): 146, (False, True): 110,
                           (True, False): 110, (True, True): 146},
            "strict-majority transition counts differ")
    changes = directions[False, True] + directions[True, False]
    print(f"XOR: each of 32 vectors has mass 16/512 under each law; "
          f"0->1: 110, 1->0: 110; paired disagreement {changes}/512 = {Fraction(changes, 512)}.")


def response_conditioning():
    """Two shared-OR slots retain a product law per complete anchor."""
    selected = Counter()
    for u0, u1 in product((0, 1), repeat=2):
        per_anchor = Counter((u0 | v1, v0 | u1) for v0, v1 in product((0, 1), repeat=2))
        for a, b in product((0, 1), repeat=2):
            left = sum(n for (x, y), n in per_anchor.items() if x == a)
            right = sum(n for (x, y), n in per_anchor.items() if y == b)
            require(per_anchor[a, b] * 4 == left * right, "original anchor lost product law")
        if u0 | u1:
            selected.update(per_anchor)
    total = sum(selected.values())
    left = Fraction(sum(n for (x, _), n in selected.items() if x), total)
    right = Fraction(sum(n for (_, y), n in selected.items() if y), total)
    both = Fraction(selected[1, 1], total)
    require((total, left, right, both) == (12, Fraction(5, 6), Fraction(5, 6), Fraction(2, 3)),
            "observed-answer conditional law changed")
    print(f"Given original (1,1): both {both}, product of marginals {left * right}.")


def first_six(run, width):
    """Read the sampled events, indexed from the first post-prefill draw."""
    lanes = run["lanes"]
    result = []
    for j, lane in enumerate(lanes):
        window = lane["events"][:width]
        require(len(window) == width, "short prefix: revisit the endpoint before interpreting this archive")
        require([e["position"] for e in window] == list(range(width)), "event positions changed")
        require(window[0]["logits"] == run["root_logits_fingerprint"], "first event is not at root logits")
        tokens = tuple(e["token"] for e in window)
        require(tokens == tuple(lane["tokens"][:width]), "event and output token IDs differ")
        if run["mode"] == "C":
            root_window = lanes[0]["events"][:width]
            require(all(e["draw_owner"] == 0 and e["new_draw"] == (j == 0)
                        and all(e[key] == root[key] for key in ("draw_word", "token", "logits"))
                        for e, root in zip(window, root_window, strict=True)), "copied-prefix provenance changed")
        else:
            require(run["mode"] == "B" and all(e["draw_owner"] == j and e["new_draw"] for e in window),
                    "own-tape draw provenance changed")
        result.append(tokens)
    return result


def matching_donors(anchor, opposite):
    donors = []
    for j, seed in enumerate(anchor["seeds"]):
        matches = [row for row in opposite if row["seeds"][j] == seed]
        require(len(matches) == 1, "same-slot donor is not unique")
        donors.append(matches[0])
    sources = {(slot, source) for row in donors for slot, source in enumerate(row["source_batches"])}
    require(len(sources) == 25, "complete donor inputs overlap")
    return donors


def paired_projections(actual, reference, width):
    def project(lanes):
        prefix = tuple(tuple(lane["tokens"][:width]) for lane in lanes)
        labels = tuple(parsed(lane) for lane in lanes)
        return prefix, labels, reported_rule(labels, True, 3)

    before, after = project(actual), project(reference)
    return tuple(a != b for a, b in zip(before, after, strict=True))


def check_archive():
    manifest = json.loads((HERE / "manifest.json").read_bytes())
    for name in ("schedule.json", "serial.jsonl.gz"):
        require(sha256((HERE / name).read_bytes()).hexdigest() == manifest[name], f"hash mismatch in {name}")
    schedule = json.loads((HERE / "schedule.json").read_bytes())["runs"]["serial"]
    meta, records = read_runs("serial", schedule)
    width = meta["prefix_tokens"]
    require(meta["model_id"] == "qwen25-3b" and meta["decode_mode"] == "serial"
            and meta["slot_count"] == 5 and width == 6, "unexpected archive configuration")
    tokens = {key: first_six(run, width) for key, run in records.items()}
    first_ids = {lane[0] for lanes in tokens.values() for lane in lanes}
    min_events = min(len(lane["events"]) for run in records.values() for lane in run["lanes"])
    print(f"Across {len(records) * 5} recorded lanes the shortest trace has {min_events} sampled events; "
          f"observed first-token IDs: {sorted(first_ids)}.")
    by_item = defaultdict(list)
    for row in schedule:
        by_item[row["id"]].append(row)

    lengths = (1, width)
    keys = [(n, mode) for n in lengths for mode in "BC"]
    totals, affected = Counter(), Counter()
    print("Serial Qwen GSM8K. Counts are five-slot unanimity (actual/reference).")
    print("question         anchors   1 token: B    C      6 tokens: B    C")
    anchors = 0
    joint_counts = {mode: Counter() for mode in "BC"}
    for item, rows in by_item.items():
        require([r["placement"] for r in rows] == ["ordinary"] * 5 + ["remix"] * 5,
                "registered row families changed")
        counts, changes = Counter(), Counter()
        for i, row in enumerate(rows):
            donors = matching_donors(row, rows[5:] if i < 5 else rows[:5])
            for mode in "BC":
                actual_lanes = records[row["batch"], mode]["lanes"]
                reference_lanes = [records[d["batch"], mode]["lanes"][j] for j, d in enumerate(donors)]
                joint_counts[mode][paired_projections(actual_lanes, reference_lanes, width)] += 1
            for n, mode in keys:
                actual = [s[:n] for s in tokens[row["batch"], mode]]
                reference = [tokens[d["batch"], mode][j][:n] for j, d in enumerate(donors)]
                left, right = len(set(actual)) == 1, len(set(reference)) == 1
                counts[n, mode, "actual"] += left
                counts[n, mode, "reference"] += right
                changes[n, mode] += left != right
        totals.update(counts)
        affected.update({key: int(changes[key] > 0) for key in keys})
        anchors += len(rows)
        cells = [f'{counts[n,mode,"actual"]}/{counts[n,mode,"reference"]}' for n, mode in keys]
        print(f"{item}  {len(rows):>2}       " + "  ".join(f"{cell:>5}" for cell in cells))

    require((len(by_item), anchors) == (12, 120), "question or directional anchor count changed")
    expected = {(1, "B"): (120, 120, 0), (1, "C"): (120, 120, 0),
                (width, "B"): (39, 39, 0), (width, "C"): (120, 40, 8)}
    for n, mode in keys:
        actual, reference = totals[n, mode, "actual"], totals[n, mode, "reference"]
        require((actual, reference, affected[n, mode]) == expected[n, mode], "prefix contrast changed")
        print(f"{n} token(s), {mode}: actual {actual}/{anchors}, reference {reference}/{anchors}; "
              f"any paired disagreement on {affected[n, mode]}/{len(by_item)} questions.")


    expected_joint = {
        "B": {(False, False, False): 120},
        "C": {(False, False, False): 40, (True, False, False): 18,
              (True, True, False): 52, (True, True, True): 10},
    }
    for mode, cells in joint_counts.items():
        require(cells == expected_joint[mode], "joined prefix, label, and strict vote counts changed")
        print(f"{mode}, (prefix vector changed, parsed vector changed, decision changed): {dict(cells)}")


if __name__ == "__main__":
    xor_majority()
    response_conditioning()
    check_archive()
