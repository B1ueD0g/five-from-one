"""Replay the registered endpoint comparisons from saved native records."""
from collections import Counter, defaultdict
from decimal import Decimal
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re

from check_seed_map import check_seed_map

HERE = Path(__file__).resolve().parent
PATTERN = re.compile(r"Final: (-?[0-9]+(?:\.[0-9]+)?)")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parsed(lane):
    if lane["terminal"] not in {"eog", "cap", "answer_line"}:
        return None
    try:
        lines = bytes.fromhex(lane["text_hex"]).decode("utf-8").split("\n")
    except UnicodeDecodeError:
        return None
    for i, line in enumerate(lines):
        delimit = i < len(lines) - 1
        content = line.rstrip("\r").strip(" \t") if delimit else line.strip(" \t")
        if content.startswith("Final:"):
            match = PATTERN.fullmatch(content)
            if match and (delimit or lane["terminal"] == "eog"):
                return Decimal(match[1])
            break
    return None


def read_runs(name, schedule, reverse=False):
    order = reversed(schedule) if reverse else schedule
    rows = {(row["batch"], mode): row for row in order for mode in row["arm_order"]}
    with gzip.open(HERE / f"{name}.jsonl.gz", "rt") as stream:
        info = json.loads(next(stream))
        values = [json.loads(line) for line in stream]
    pairs = [(row["batch"], row["mode"]) for row in values]
    require(info["kind"] == "metadata" and pairs == list(rows), f"{name} arm order or coverage")
    runs = {}
    for pair, value in zip(pairs, values, strict=True):
        row = rows[pair]
        require(all(row[field] == value[field] for field in ("seeds", "item_id", "source_batches")),
                f"{name} tape identity drift")
        require([x["slot"] for x in value["lanes"]] == list(range(5)), f"{name} slot order")
        if pair[1] == "B":
            require(all(event["draw_owner"] == lane["slot"] for lane in value["lanes"]
                        for event in lane["events"]), f"{name} mode B owner")
        runs[pair] = value
    return info, runs


def events(lane):
    return [{key: value for key, value in event.items() if key != "forward"} for event in lane["events"]]


def delta(a, b):
    return (events(a) != events(b),
            any(a[k] != b[k] for k in ("tokens", "terminal", "text_hex")),
            parsed(a) != parsed(b))


def fixed(a, b):
    results = {mode: [0, 0, 0, 0] for mode in "BC"}
    for (batch, mode), left in a.items():
        right = b[batch, mode]
        require(left["prompt_tokens"] == right["prompt_tokens"], "fixed prompt changed")
        counts = results[mode]
        for x, y in zip(left["lanes"], right["lanes"], strict=True):
            for n, flag in enumerate(delta(x, y)):
                counts[n] += flag
            counts[3] += 1
    require(results == {mode: [0, 0, 0, 150] for mode in "BC"}, "fixed-tape counts changed")
    return results


def outer(records, roster):
    totals = {mode: [0, 0, 0, 0] for mode in "BC"}
    first = []
    groups = defaultdict(list)
    for row in roster:
        groups[row["id"]].append(row)
    for item, rows in groups.items():
        require(len(rows) == 10, "incomplete tape array")
        for i, anchor in enumerate(rows[:5]):
            for slot in range(5):
                donor = rows[5 + (i + slot) % 5]
                require(all((a == b) == (j == slot) for j, (a, b) in enumerate(
                    zip(anchor["seeds"], donor["seeds"], strict=True))), "wrong donor")
                for mode in "BC":
                    a = records[anchor["batch"], mode]["lanes"][slot]
                    b = records[donor["batch"], mode]["lanes"][slot]
                    d = delta(a, b)
                    for n, flag in enumerate(d):
                        totals[mode][n] += flag
                    totals[mode][3] += 1
                    if mode == "B" and d[1]:
                        k, x, y = next((k, x, y) for k, (x, y) in enumerate(zip(events(a), events(b)))
                                       if x["token"] != y["token"])
                        require(all(x[s] == y[s] for s in ("draw_owner", "draw_word", "new_draw"))
                                and x["logits"] != y["logits"], "first numerical divergence changed")
                        first.append((item, anchor["batch"], donor["batch"], slot, k))
    expected = [("GSM8K-test-0511", 65001, 65008, 2, 34),
                ("GSM8K-test-0511", 65002, 65007, 0, 38),
                ("GSM8K-test-0511", 65004, 65009, 0, 39)]
    require(sorted(first) == expected, "wrong first-divergence witness")
    require(totals == {"B": [18, 3, 3, 75], "C": [64, 23, 23, 75]}, "companion counts changed")
    return totals


def release(labels):
    require(None not in labels, "vote witness contains invalid label")
    answer, count = Counter(labels).most_common(1)[0]
    require(count >= 3, "witness no longer has a majority")
    return answer


def vote_example(records, roster, gold):
    item = "GSM8K-test-0680"
    rows = [row for row in roster if row["id"] == item]
    anchor = next(row for row in rows if row["batch"] == 65063)
    donors = []
    for slot in range(5):
        matches = [row for row in rows[5:] if row["seeds"][slot] == anchor["seeds"][slot]]
        require(len(matches) == 1, "wrong anchor matching")
        donors.append(matches[0])
    require(len({(j, r) for row in donors for j, r in enumerate(row["source_batches"])}) == 25,
            "donor tuples overlap")
    left = [parsed(l) for l in records[65063, "C"]["lanes"]]
    right = [parsed(records[d["batch"], "C"]["lanes"][j]) for j, d in enumerate(donors)]
    require(left == [576]*5 and right == [576, 16, 16, 16, 16], "vote vectors changed")
    require(release(left) == Decimal(gold[item]) == 576 and release(right) == 16, "release changed")


def reported_rule(labels, all_valid, threshold):
    valid = [label for label in labels if label is not None]
    if all_valid and len(valid) != 5:
        return None
    top = Counter(valid).most_common(2)
    if not top or top[0][1] < threshold or len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def compare_rules(records, roster, profile):
    rules = ((True, 3), (False, 3), (True, 4), (False, 1))
    by_item = defaultdict(list)
    for row in roster:
        by_item[row["id"]].append(row)
    differences = {mode: [0]*len(rules) for mode in "BC"}
    affected = {mode: [0]*len(rules) for mode in "BC"}
    pairs = 0
    for rows in by_item.values():
        changed = {mode: [0]*len(rules) for mode in "BC"}
        for i, row in enumerate(rows):
            donors = ([rows[5 + (i + j) % 5] for j in range(5)] if i < 5 else
                      [rows[(i - 5 - j) % 5] for j in range(5)])
            require(all(row["seeds"][j] == donor["seeds"][j]
                        for j, donor in enumerate(donors)), "inverse rule contrast lost anchor")
            require(len({(j, source) for d in donors for j, source in enumerate(d["source_batches"])}) == 25,
                    "rule contrast overlaps complete donor tapes")
            for mode in "BC":
                observed = [parsed(lane) for lane in records[row["batch"], mode]["lanes"]]
                product = [parsed(records[d["batch"], mode]["lanes"][j]) for j, d in enumerate(donors)]
                for n, rule in enumerate(rules):
                    changed[mode][n] += reported_rule(observed, *rule) != reported_rule(product, *rule)
            pairs += 1
        for mode in "BC":
            for n, val in enumerate(changed[mode]):
                differences[mode][n] += val
                affected[mode][n] += val > 0
    expected = {
        "serial": (120, [10, 10, 11, 22], [3, 3, 3, 6]),
        "parallel": (30, [2, 2, 0, 8], [1, 1, 0, 2]),
    }[profile]
    require((pairs, differences["C"], affected["C"]) == expected and not any(differences["B"]),
            "registered voting-rule sensitivity differs")
    print(profile, "voting-rule differences", differences, "of", pairs,
          "directional pairs; affected questions", affected)


def main():
    manifest = json.loads((HERE / "manifest.json").read_bytes())
    require(set(manifest) == {"baseline.jsonl.gz", "repeat.jsonl.gz", "reverse.jsonl.gz", "serial.jsonl.gz",
                              "gold.json", "schedule.json"}, "manifest scope changed")
    for name, digest in manifest.items():
        require(sha256((HERE / name).read_bytes()).hexdigest() == digest, f"hash mismatch in {name}")
    schedule = json.loads((HERE / "schedule.json").read_bytes())["runs"]
    check_seed_map([row for rows in schedule.values() for row in rows])
    gold = json.loads((HERE / "gold.json").read_bytes())
    meta, original = read_runs("baseline", schedule["parallel"])
    print("Original companion differences [event, terminal, label, denominator]:", outer(original, schedule["parallel"]))
    for name in ("repeat", "reverse"):
        profile, new = read_runs(name, schedule["parallel"], reverse=name == "reverse")
        require(profile == meta, "parallel metadata changed")
        print(name, "fixed:", fixed(original, new), "companion:", outer(new, schedule["parallel"]))
    _, serial = read_runs("serial", schedule["serial"])
    vote_example(serial, schedule["serial"], gold)
    compare_rules(serial, schedule["serial"], "serial")
    compare_rules(original, schedule["parallel"], "parallel")
    print("Verified the 0680 vote witness (576 vs. 16) and all three B first-divergence locations.")


if __name__ == "__main__":
    main()
