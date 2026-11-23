"""Recompute published table counts from native numeric records and archived extracts."""
from collections import Counter, defaultdict
from decimal import Decimal
import json

from check_estimands import matching_donors
from verify import HERE, parsed, read_runs, reported_rule, require


def read(name):
    return json.loads((HERE / name).read_bytes())


def count_decisions(pairs, gold):
    counts = Counter()
    for actual, reference in pairs:
        left = reported_rule(actual, True, 3)
        right = reported_rule(reference, True, 3)
        both_valid = None not in actual and None not in reference
        counts["pairs"] += 1
        counts["changed"] += left != right
        counts["both_valid"] += both_valid
        counts["both_valid_changed"] += both_valid and left != right
        counts["correctness_flips"] += None not in (left, right) and ((left == gold) != (right == gold))
        for label, decision in (("actual", left), ("reference", right)):
            counts[f"{label}_correct"] += decision == gold
            counts[f"{label}_wrong"] += decision is not None and decision != gold
    counts["affected_questions"] += counts["changed"] > 0
    return counts


def numeric_counts():
    schedule = read("schedule.json")["runs"]["serial"]
    gold = read("gold.json")
    _, records = read_runs("serial", schedule)
    by_item = defaultdict(list)
    for row in schedule:
        by_item[row["id"]].append(row)
    summaries = {mode: Counter() for mode in "BC"}
    for item, rows in by_item.items():
        for mode, totals in summaries.items():
            pairs = []
            for i, row in enumerate(rows):
                donors = matching_donors(row, rows[5:] if i < 5 else rows[:5])
                current = records[row["batch"], mode]["lanes"]
                products = [records[d["batch"], mode]["lanes"][j] for j, d in enumerate(donors)]
                pairs.append((list(map(parsed, current)), list(map(parsed, products))))
            totals.update(count_decisions(pairs, Decimal(gold[item])))
    return summaries


def arc_counts(rows):
    summaries = {mode: Counter() for mode in "BC"}
    for item in rows:
        require(item["status"] == "ACCEPT", "ARC extract has unavailable evidence")
        for mode, totals in summaries.items():
            records = item["modes"][mode]
            require([r["family"] for r in records] == ["ordinary"] * 5 + ["remix"] * 5,
                    "ARC donor family order changed")
            actual = [r["labels"] for r in records]
            require(all(len(labels) == 5 and set(labels) <= {*"ABCD", None} for labels in actual),
                    "ARC parsed label extract is malformed")
            products = [[actual[5 + (i + j) % 5][j] for j in range(5)] for i in range(5)]
            products += [[actual[(i - j) % 5][j] for j in range(5)] for i in range(5)]
            totals.update(count_decisions(zip(actual, products, strict=True), item["gold"]))
    return summaries


def verify_publish_counts():
    arc = read("arc_labels.json")
    qwen = arc["qwen_arc_initial"]["questions"] + arc["qwen_arc_holdout"]["questions"]
    ministral = arc["ministral_arc"]["questions"]
    mlx = arc["qwen_mlx_arc_subset"]["questions"]
    cases = {
        "GSM8K serial": (numeric_counts(), (120, 10, 114, 9, 2, 3, 22, 20, 49, 49)),
        "ARC Qwen": (arc_counts(qwen), (280, 55, 213, 12, 4, 13, 204, 190, 34, 32)),
        "ARC Ministral": (arc_counts(ministral), (200, 35, 153, 4, 3, 11, 154, 154, 15, 13)),
        "ARC MLX": (arc_counts(mlx), (60, 2, 1, 0, 0, 1, None, None, None, None)),
    }
    fields = ("pairs", "changed", "both_valid", "both_valid_changed", "correctness_flips",
              "affected_questions", "actual_correct", "reference_correct", "actual_wrong", "reference_wrong")
    for name, (actual, expected) in cases.items():
        result = {field: actual["C"][field] for field in fields}
        print(name, result)
        require(actual["B"]["changed"] == 0 and
                all(v is None or result[k] == v for k, v in zip(fields, expected, strict=True)),
                f"{name}: published result differs from artifact")
    require(len({r["id"] for r in qwen}) == 28 and
            len({r["id"] for r in qwen} & {r["id"] for r in ministral}) == 8 and
            {r["id"] for r in mlx} <= {r["id"] for r in qwen},
            "ARC roster identities or overlaps changed")


def verify_policy_counts():
    questions = read("policy_labels.json")["questions"]
    require(len({q["id"] for q in questions}) == len(questions) == 373
            and all(q["record_status"] == {"B": "ok", "C": "ok"} for q in questions),
            "GSM8K policy roster is incomplete")
    counts = Counter()
    for q in questions:
        for mode in "BC":
            labels = [None if x is None else Decimal(x) for x in q["modes"][mode]]
            decision = reported_rule(labels, True, 3)
            counts[mode, "correct"] += decision == Decimal(q["gold"])
            counts[mode, "wrong"] += decision is not None and decision != Decimal(q["gold"])
    require(dict(counts) == {("B", "correct"): 80, ("B", "wrong"): 104,
                             ("C", "correct"): 85, ("C", "wrong"): 127},
            "GSM8K historical policy count mismatch")
    print("GSM8K historical policy", dict(counts))


def verify_costs():
    expected = {"full_width": {"baseline": (8, 2815, 1087), "packed": (4, 1872, 720)},
                "selective": {"baseline": (8, 1094, 518), "packed": (4, 1094, 518)}}
    for interface, strategies in read("cost_rows.json").items():
        for name, capture in strategies.items():
            rows = capture["rows"]
            anchors = {r["item_id"] for r in rows}
            require(len(anchors) == 4 and len(rows) == capture["metadata_count"], "cost roster changed")
            metrics = (len(rows) - len(anchors), sum(r["decoded_tokens"] for r in rows),
                       sum(r["decode_calls"] for r in rows))
            require(metrics == expected[interface][name], "published cost total changed")
            print(interface, name, metrics)


if __name__ == "__main__":
    verify_publish_counts()
    verify_policy_counts()
    verify_costs()
