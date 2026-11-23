# Five from One? A Replay Audit of Batched LLM Sampling

Research artifact for the manuscript by Songbo Bu. It contains the replay audit, preserved capture records, and scripts that recover the main decision and work counts from those records or their documented extracts.

## Recompute the results

Use Python 3.10 or later from the repository root. The audit requires only the standard library. It runs offline with the data included in the repository.

```sh
python3 -B verify_release.py
```

| Evidence | What the check reconstructs |
| --- | --- |
| [GSM8K captures and schedules](artifact/README.md) | The 0680 vote witness, serial and parallel comparisons, two fresh-process checks, fixed-rule results, first divergence records, prefix diagnostics, alignment counts and exact finite controls |
| [ARC and historical GSM8K label extracts](artifact/README.md#derived-extracts) | Strict decision changes, all-valid pairs, correct-to-incorrect and incorrect-to-correct switches, and historical B/C release counts |
| [Work-count extracts](artifact/README.md#derived-extracts) | Additional requests, model token rows and decode calls for the full-width and selective interfaces |
| [Figure sources and snapshots](figures/README.md) | The four manuscript figures, with Python inputs and PDF exports |
| [Native replay application](native/README.md) | C++ source, registered GSM8K inputs, model and runtime pins, and macOS build and capture commands |

The main numeric panel uses 12 GSM8K questions. Its three-question parallel panel reuses the registered input tapes for those questions. The two donor directions reuse calls. The ARC extracts preserve each parsed slot label, including invalid slots. They support offline rescoring given those labels. The historical 373-question policy comparison uses a separate roster and one original group per mode and question.

The offline checks assume a trusted capture producer. Backend or hardware changes can affect the native traces. The view in the matching analysis consists of the recorded question and direction strata, so its extrema describe the observed panel.

## Attribution

Use the repository's [citation metadata](CITATION.cff) and review [license and data provenance](licenses/README.md) before reuse.
