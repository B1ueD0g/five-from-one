"""Draw the release robustness figure from frozen labels and aggregate counts."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "rule_figure_counts.json"
RULES = {
    "strict_three": (True, 3, "Strict\nthree"),
    "quorum_three": (False, 3, "Quorum\nthree"),
    "strict_four": (True, 4, "Strict\nfour"),
    "unique_plurality": (False, 1, "Unique\nplurality"),
}
COHORTS = ("ARC · Qwen2.5 (28 questions)", "ARC · Ministral (20 questions)",
           "GSM8K · Qwen2.5 (12 questions)")
CAUSES = ("Invalid slot in abstaining group", "All valid, no majority", "Different answer")
INK, SUB, FRAME, PALE = "#182b39", "#364b5b", "#ccd6de", "#f0f5f8"
BLUE, TEAL, AMBER, GRAY, WHITE = "#285f83", "#26746f", "#a35520", "#9babb8", "#ffffff"
CAUSE_STYLE = ((GRAY, "#667484", None), ("#f8e8d4", AMBER, "///"), (TEAL, TEAL, None))
CELL_MAP = LinearSegmentedColormap.from_list("counts", ("#edf5f8", "#5b99b3", "#164561"))


def read_json(path):
    return json.loads(path.read_bytes())


def plot_decomposition(fig, counts):
    fig.text(.023, .975, "a   Why strict release changes", fontsize=10.3, weight="bold",
             va="top", color=INK)
    chart = fig.add_axes((.415, .707, .543, .179))
    handles = [Patch(facecolor=fc, edgecolor=ec, hatch=pattern, label=label)
               for (fc, ec, pattern), label in zip(CAUSE_STYLE, CAUSES)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.021, .937),
               frameon=False, ncol=3, fontsize=8.3, handlelength=1.18,
               handletextpad=.38, columnspacing=1.05, borderaxespad=0)
    chart.set_axisbelow(True)
    chart.set(xlim=(0, .248), ylim=(2.5, -.5), xticks=(0, .05, .1, .15, .2))
    chart.set_yticks(range(3), COHORTS)
    chart.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    chart.tick_params(axis="both", length=0, labelsize=8.4)
    chart.grid(axis="x", color=FRAME, linewidth=.57)
    chart.axvline(0, color=SUB, linewidth=.57)
    for row, (segments, denominator, _) in enumerate(counts):
        left = 0
        for n, (fill, edge, hatch) in zip(segments, CAUSE_STYLE):
            width = n / denominator
            if n:
                chart.barh(row, width, height=.46, left=left, color=fill,
                           edgecolor=edge, hatch=hatch, linewidth=.72)
            left += width
        chart.text(left + .007, row, f"{sum(segments)}/{denominator}", color=INK,
                   va="center", ha="left", fontsize=8.4)
    chart.set_xlabel("Pairs with a changed release", labelpad=2, fontsize=8.6)


def plot_matrix(fig, counts):
    fig.text(.023, .635, "b   Which GSM8K questions change across voting rules",
             fontsize=10.3, weight="bold", va="top", color=INK)
    matrix = fig.add_axes((.023, .074, .954, .518))
    matrix.set(xlim=(0, 10.85), ylim=(15.1, 0))
    matrix.axis("off")
    starts = {"serial": 1.52, "parallel": 6.26}
    plot_profiles = counts["panels"]
    upper = max(n for p in plot_profiles.values() for row in p["cells"] for n in row if n is not None)
    width, height = .965, .78
    row_start, row_stride = 2.48, .852
    matrix.text(.04, 1.58, "Test ID", ha="left", va="center", fontsize=8.5, weight="bold")
    for profile, title in (("serial", "Serial"), ("parallel", "Parallel subset")):
        left = starts[profile]
        data = plot_profiles[profile]
        matrix.text(left + 1.98, .36, f"{title} · {data['n_pairs']} pairs",
                    ha="center", va="center", fontsize=9.1, weight="bold", color=INK)
        for col, (_, _, label) in enumerate(RULES.values()):
            matrix.text(left + col + width/2, 1.42, label, ha="center", va="center",
                        fontsize=8.3, color=SUB, linespacing=1.01)
        for row, values in enumerate(data["cells"]):
            y = row_start + row_stride * row
            for col, n in enumerate(values):
                x = left + col
                if n is None:
                    fill, edge, pattern, label, foreground = "#f2f4f5", FRAME, "///", "–", SUB
                elif n == 0:
                    fill, edge, pattern, label, foreground = WHITE, FRAME, None, "0", SUB
                else:
                    intensity = .18 + .78*n/upper
                    fill, edge, pattern, label = CELL_MAP(intensity), BLUE, None, str(n)
                    foreground = WHITE if intensity > .57 else INK
                matrix.add_patch(Rectangle((x, y-height/2), width, height, facecolor=fill,
                                            edgecolor=edge, hatch=pattern, linewidth=.48))
                matrix.text(x+width/2, y, label, ha="center", va="center", fontsize=8.4,
                            color=foreground, weight="bold" if n else "normal")
        total_y = 13.03
        matrix.plot([left, left+4-.04], [12.61, 12.61], color=SUB, lw=.62)
        for col, n in enumerate(data["totals"]):
            matrix.text(left + col + width/2, total_y, str(n), ha="center", va="center",
                        fontsize=9, weight="bold", color=INK)

    for row, id_ in enumerate(counts["question_ids"]):
        matched = plot_profiles["parallel"]["cells"][row][0] is not None
        matrix.text(.04, row_start + row_stride*row, id_.rsplit("-", 1)[-1], ha="left",
                    va="center", fontsize=8.5, color=INK, weight="bold" if matched else "normal")
    matrix.text(.04, 13.03, "Total", ha="left", va="center", fontsize=8.5, weight="bold")
    matrix.text(.04, 14.05, "Cell = changed pairs out of 10.    –  Not run in parallel.",
                ha="left", va="center", fontsize=8.4, color=SUB)
    matrix.text(.04, 14.87, "All rules reuse the same parsed answers within each profile.",
                ha="left", va="center", fontsize=8.4, color=SUB)


def draw(counts):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.6, "text.color": INK,
        "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "release_robustness",
        "axes.edgecolor": FRAME, "axes.labelcolor": SUB,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "xtick.color": SUB, "ytick.color": INK, "hatch.linewidth": .45,
    })
    fig = plt.figure(figsize=(6.15, 5.82), facecolor=WHITE)
    plot_decomposition(fig, counts["release_counts"])
    plot_matrix(fig, counts)
    fig.savefig(HERE / "release_robustness.pdf", metadata={
        "Title": "Release robustness", "Author": None, "CreationDate": None, "ModDate": None,
    })
    fig.savefig(HERE / "release_robustness.svg", metadata={
        "Title": "Release robustness", "Date": None,
    })
    plt.close(fig)


if __name__ == '__main__':
    draw(read_json(SNAPSHOT))
