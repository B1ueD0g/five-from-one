# Rebuild the figures

Run these commands from the repository root with a local Python environment. The drawing scripts were checked with Matplotlib 3.11.0.

```sh
python3 -m pip install matplotlib==3.11.0
python3 -B figures/draw_motivation.py
python3 -B figures/draw_figures.py
python3 -B figures/draw_rule_figure.py
```

The scripts rewrite their PDFs and SVGs next to the figure sources. Run them in a copy to retain the packaged bytes. SVG element identifiers and renderer metadata may vary between installations.

| PDF | Plot input |
| --- | --- |
| `motivation.pdf` | `motivation_counts.json`, the numeric vote witness and repeated execution comparison |
| `design.pdf` | Cyclic registration and donor assembly encoded in the drawing script |
| `results.pdf` | `reported_counts.json`, the serial and parallel comparison counts and first-divergence records |
| `release_robustness.pdf` | `rule_figure_counts.json`, the abstention and correctness decomposition and consumer-rule counts |

The saved plot inputs identify their archived sources by digest. The commands redraw from snapshots. Use the [numerical checks](../artifact/README.md) for reconstruction from archived traces and extracts.
