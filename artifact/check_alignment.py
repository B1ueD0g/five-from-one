"""Remove anchor alignment by every nonidentity rotation of a fixed five-row family.

Check every row bijection within each question and direction against the sharp
histogram-only disagreement bounds. All recodings reuse the same native records.
Run with python3 -B artifact/check_alignment.py from the manuscript directory.
"""
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import permutations
import json

from verify import HERE, parsed, read_runs, reported_rule, require


def disagreement_bounds(actual, product):
    """Check the smallest and largest disagreement counts over every row bijection."""
    n = len(actual)
    a, b = Counter(actual), Counter(product)
    require(n == len(product) == 5, "expected two five-row families")
    most_matches = sum((a & b).values())
    least_matches = max(0, max(total - n for total in (a + b).values()))
    lower, upper = n - most_matches, n - least_matches
    observed = [sum(x != y for x, y in zip(actual, perm, strict=True))
                for perm in permutations(product)]
    require((min(observed), max(observed)) == (lower, upper), "histogram bound differs from enumeration")
    return lower, upper


def analyze_family(anchors, carriers, records, mode):
    """Return changes for the five rotations and bounds for all row bijections."""
    actual = []
    product = []
    own_tapes = []
    for anchor in anchors:
        donors = []
        for slot in range(5):
            matching = [row for row in carriers if row["seeds"][slot] == anchor["seeds"][slot]]
            require(len(matching) == 1, "ambiguous or missing same-tape donor")
            donors.append(matching[0])
        support = {(slot, source) for row in donors for slot, source in enumerate(row["source_batches"])}
        require(len(support) == 25, "overlapping donor inputs within one product vector")
        original_lanes = records[anchor["batch"], mode]["lanes"]
        product_lanes = [records[row["batch"], mode]["lanes"][slot] for slot, row in enumerate(donors)]
        actual.append(reported_rule([parsed(lane) for lane in original_lanes], True, 3))
        product.append(reported_rule([parsed(lane) for lane in product_lanes], True, 3))
        own_tapes.append([row["seeds"][slot] for slot, row in enumerate(donors)])

    changed = []
    for shift in range(5):
        permutation = [(i + shift) % 5 for i in range(5)]
        rematched = [product[index] for index in permutation]
        require(Counter(rematched) == Counter(product), "product endpoint histogram changed")
        for i, index in enumerate(permutation):
            require(all((a == b) == (shift == 0) for a, b in
                        zip(anchors[i]["seeds"], own_tapes[index], strict=True)), "unexpected tape match")
        changed.append(sum(a != b for a, b in zip(actual, rematched, strict=True)))
    # Retain the identity of each original decision; discard which product row belongs to it.
    product_counts = Counter(product)
    row_status = Counter()
    for value, aligned_value in zip(actual, product, strict=True):
        matches = product_counts[value]
        if matches == 0:
            row_status["always changes"] += 1
        elif matches == len(product):
            row_status["always agrees"] += 1
        else:
            row_status["unresolved"] += 1
            row_status["unresolved and changed when aligned"] += value != aligned_value
    return changed, disagreement_bounds(actual, product), row_status


def main():
    manifest = json.loads((HERE / "manifest.json").read_bytes())
    for name in ("serial.jsonl.gz", "schedule.json"):
        require(sha256((HERE / name).read_bytes()).hexdigest() == manifest[name], f"hash mismatch in {name}")
    schedule = json.loads((HERE / "schedule.json").read_bytes())["runs"]["serial"]
    _, records = read_runs("serial", schedule)
    by_item = defaultdict(list)
    for row in schedule:
        by_item[row["id"]].append(row)
    require(len(by_item) == 12 and all(len(rows) == 10 for rows in by_item.values()), "roster changed")

    print("mode offset forward/60 reverse/60 total/120 affected_questions/12")
    for mode in ("B", "C"):
        forward = Counter()
        reverse = Counter()
        affected = Counter()
        lower_total = upper_total = 0
        excess = []
        statuses = Counter()
        for item, rows in by_item.items():
            fwd, fwd_bounds, fwd_status = analyze_family(rows[:5], rows[5:], records, mode)
            rev, rev_bounds, rev_status = analyze_family(rows[5:], rows[:5], records, mode)
            statuses.update(fwd_status)
            statuses.update(rev_status)
            for direction, aligned, (lower, upper) in (
                    ("forward", fwd[0], fwd_bounds), ("reverse", rev[0], rev_bounds)):
                lower_total += lower
                upper_total += upper
                if aligned > lower:
                    excess.append((item, direction, lower, aligned))
            for shift in range(5):
                forward[shift] += fwd[shift]
                reverse[shift] += rev[shift]
                affected[shift] += (fwd[shift] + rev[shift] > 0)
        expected = {"B": [0, 29, 24, 24, 29], "C": [10, 27, 26, 22, 29]}[mode]
        expected_questions = {"B": [0, 5, 5, 5, 5], "C": [3, 6, 6, 6, 6]}[mode]
        require([forward[s] + reverse[s] for s in range(5)] == expected, "decision counts changed")
        require([affected[s] for s in range(5)] == expected_questions, "question counts changed")
        for shift in range(5):
            print(mode, shift, forward[shift], reverse[shift], forward[shift] + reverse[shift], affected[shift])
        fields = ("always changes", "always agrees", "unresolved", "unresolved and changed when aligned")
        expected_status = (0, 75, 45, 0) if mode == "B" else (4, 81, 35, 6)
        require(tuple(statuses[f] for f in fields) == expected_status, "named-row ambiguity changed")
        print(f"{mode}, original decisions retained: {dict(statuses)}")
        totals = (lower_total, forward[0] + reverse[0], upper_total)
        require(totals == {"B": (0, 0, 32), "C": (8, 10, 32)}[mode], "pooled counts changed")
        expected_excess = [] if mode == "B" else [("GSM8K-test-0070", "forward", 2, 4)]
        require(excess == expected_excess, "unexpected location above the minimum")
        print(f"{mode}: minimum/aligned/maximum across question-direction strata = "
              f"{totals[0]}/{totals[1]}/{totals[2]} (120 pairs)")
        for item, direction, lower, aligned in excess:
            print(f"  {item} {direction}: minimum {lower}, aligned {aligned}.")
    print("Product endpoint histograms match exactly for every item, direction, mode and rotation.")


if __name__ == "__main__":
    main()
