# Five from One? Anonymous numerical evidence

Run from the extracted directory with Python 3.10 or later.

```sh
python3 verify.py
```

The four gzip files contain the exact captured JSONL bytes for the original serial run, original parallel run, and two separate parallel process repeats. `schedule.json` registers the row order, model identifier, and tape identities. `gold.json` contains public GSM8K answer labels. SHA-256 hashes identify the bundled input bytes.

The verifier checks row and slot identities, replay denominators, stored draw owners under mode B, and the terminal, event, and parsed-label counts of the fixed-tape and companion-tape comparisons. It also rebuilds the five votes and validates their complete donor input disjointness for GSM8K test question 0680. The verifier also reconstructs the four consumer-rule contrasts from the original serial and parallel capture, in both directions. The reverse repeat keeps the mode order within each row and reverses the sequence of rows.

This is offline analysis of instrumented execution records. The capture producer is trusted. Model weights, a hardware attestation, and the native inference environment are outside this bundle.

Run `python3 -B check_estimands.py` for the paired-majority counterexample and the post hoc token-level check. The exact check enumerates all 512 equally likely assignments of five anchor bits and four donor roots. Both five-bit vectors are uniform, although their paired majorities disagree on 220 assignments.

The archive check uses two projections chosen from the decoder design: the first sampled token and the complete six-token shared prefix recorded in the capture metadata. Every serial lane reaches six draws. It checks event positions, token IDs, draw ownership, and the copy flags, then counts five-slot unanimity using all ten directional anchors per question. It constructs the reference from the unique same-slot donor in the opposite family and checks disjoint complete donor inputs. The printed rows give the actual and reference counts separately for B (deterministic prefill reuse) and C (sampled-prefix reuse). The two directions reuse the same tape array. The 120 pairs describe these 12 captured questions.

Run `python3 -B check_alignment.py` to reconstruct the aligned five-vote comparison and all four nonzero rotations within each question and donor direction. The script checks the complete product-answer histogram, including abstentions, and the own-slot tape matches for every rotation. The rotations reuse all observations.

The estimand check also enumerates the two-slot shared-OR example and joins six-token prefix, parsed-label and strict-decision readouts on each recorded anchor. The alignment check enumerates all 120 possible bijections within each question and direction, and compares their extrema to counts derived directly from the two decision histograms. The script also retains each original decision and reports which named rows must change, must agree or are ambiguous after hiding their product matches. These permutations generate no new model observations.

Run `python3 -B check_seed_map.py` for within-question source/seed uniqueness and a negative control with two sources assigned the same 64-bit seed. The main verifier also runs the identity check against the registered schedules. Distinct seeds do not certify statistical independence.

## Derived extracts

`arc_labels.json` records the question ID, gold choice, registration status and the five parsed choices for each original and remixed batch in the two Qwen cohorts, the matched Qwen MLX cohort, and the Ministral cohort. The checker independently constructs each product row from the donor family and rescores the strict five-vote decision. It preserves `null` for ineligible parsed slots. It starts from archived parsed labels. The full ARC output texts and the benchmark question texts are held separately.

`policy_labels.json` records the two observed parsed answer vectors and their capture status for each of the 373 historical GSM8K policy questions. `cost_rows.json` retains the recorded work counters by request for both interfaces and both replay strategies. The check recomputes counts from these extracts. Their `source_sha256` fields identify the full project archives from which the extracts were derived. The root manifest identifies the distributed extract bytes.

Run `python3 -B artifact/check_tables.py` from the repository root to recompute the strict-release and work counts with these inputs.
