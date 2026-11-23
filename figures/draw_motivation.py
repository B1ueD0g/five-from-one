"""Render the editable motivation figure from the frozen archived observations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
FONT = "DejaVu Sans"
W, H = 468, 287
INK = "#16252c"
MUTED = "#445763"
LINE = "#cad6dc"
BLUE = "#267080"
BLUE_S = "#ebf5f6"
AMBER = "#a35425"
AMBER_S = "#fbecdc"
SOFT = "#f6f8f9"
WHITE = "#ffffff"


def text(ax, x, y, message, size=9.2, weight="normal", align="left", color=INK, zorder=5):
    return ax.text(x, y, message, color=color, fontfamily=FONT, size=size,
                   weight=weight, ha=align, va="center", zorder=zorder)


def rounded(ax, x, y, w, h, fill, stroke="none", radius=3.5, lw=0.7, zorder=2):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=fill, edgecolor=stroke, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box


def rule(ax, x1, x2, y, color=LINE, lw=0.66, dash="solid"):
    ax.plot([x1, x2], [y, y], lw=lw, color=color, linestyle=dash, zorder=1)


def panel_heading(ax, tag, y, heading):
    text(ax, 9, y, tag, 12, "bold")
    text(ax, 25, y, heading, 12.2, "bold")


def draw(evidence):
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none",
                        "font.family": FONT, "mathtext.fontset": "dejavusans",
                        "svg.hashsalt": "FiveFromOne-figure1"})
    fig = plt.figure(figsize=(W/72, H/72), facecolor=WHITE)
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, W), ylim=(H, 0))
    ax.set_axis_off()
    witness = evidence["witness"]
    control = evidence["control"]
    short_id = witness["item"].split("-")[-1]

    panel_heading(ax, "a", 11, "One question, two voting outcomes")
    text(ax, 460, 12, f"GSM8K test {short_id}  ·  Gold {witness['reference']}",
         8.9, align="right", color=MUTED)
    rule(ax, 8, 460, 25)

    text(ax, 12, 47, "Slot", 8.8, color=MUTED)
    text(ax, 92, 39, "Observed group", 10.2, "bold", "center")
    text(ax, 92, 53, "Shared-prefix run", 8.8, align="center", color=MUTED)
    text(ax, 211, 40, "Fixed slot tape", 10.2, "bold", "center")
    text(ax, 211, 53, "Same assigned stream", 8.8, align="center", color=MUTED)
    text(ax, 343, 39, "Replay product", 10.2, "bold", "center")
    text(ax, 343, 53, "One vote per donor call", 8.8, align="center", color=MUTED)
    text(ax, 447, 47, "Donor", 8.8, align="right", color=MUTED)

    for slot, (before, after, donor) in enumerate(zip(witness["original"], witness["product"],
                                                      witness["donors"], strict=True)):
        y = 70 + slot * 20
        changed = (before != after)
        text(ax, 19, y, str(slot), 9.2, align="center", color=MUTED)
        rule(ax, 119, 315, y, BLUE, lw=0.78)
        rounded(ax, 64, y-8, 55, 16, BLUE_S)
        text(ax, 91.5, y, before, 11.2, "bold", "center")
        rounded(ax, 196, y-8, 30, 16, WHITE, BLUE)
        text(ax, 211, y, rf"$u_{{{slot}}}$", 9.3, align="center")
        rounded(ax, 315, y-8, 55, 16, AMBER_S if changed else BLUE_S,
                AMBER if changed else "none")
        text(ax, 342.5, y, after, 11.2, "bold", "center")
        text(ax, 447, y, f"R{donor['row']+1}", 8.9, align="right", color=MUTED)
    rule(ax, 8, 460, 163)

    for x, vote, label, values, col, soft in (
        (17, witness["release"][0], witness["status"][0], witness["original"], BLUE, BLUE_S),
        (281, witness["release"][1], witness["status"][1], witness["product"], AMBER, AMBER_S)):
        rounded(ax, x, 172, 170, 30, soft)
        ax.plot([x, x], [174, 200], color=col, lw=2.0, solid_capstyle="round", zorder=3)
        text(ax, x+8, 182, f"Release {vote}", 13.2, "bold")
        text(ax, x+8, 194.7, f"{values.count(vote)}/5 votes  ·  {'Wrong' if label=='incorrect' else 'Correct'}",
             8.9, color=MUTED)
    text(ax, 234, 182, "All five valid", 8.8, align="center", color=MUTED)
    text(ax, 234, 194.7, "At least 3 agree", 8.8, align="center", color=MUTED)

    rule(ax, 8, 460, 213)
    panel_heading(ax, "b", 225, "Parallel decoding with slot-local draws")
    text(ax, 460, 225, f"Separate control · {control['questions']} questions",
         8.8, align="right", color=MUTED)
    rounded(ax, 9, 237, 450, 38, SOFT, LINE, radius=3.0)
    ax.plot([234, 234], [242, 270], color=LINE, lw=0.7, zorder=3)
    text(ax, 19, 247, "Replay all five tapes", 9.5, "bold")
    text(ax, 247, 247, "Keep one tape, vary the other four", 9.5, "bold")
    for x, key in ((19, "full_tape"), (247, "same_own_tape")):
        n, d = control[key]
        text(ax, x, 263, f"{n}/{d}", 13.0, "bold")
        text(ax, x+(57 if key == "full_tape" else 48), 263, "terminal sequences changed", 8.8, color=MUTED)

    fig.savefig(HERE/"motivation.pdf", metadata={
        "Title": "Five from One? Figure 1", "CreationDate": None, "ModDate": None})
    fig.savefig(HERE/"motivation.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == '__main__':
    draw(json.loads((HERE / 'motivation_counts.json').read_bytes()))
