# Sweep rerun conclusions (30-seed stability)

Date: 2025-12-14

This note summarizes what we learn from **rerunning selected “top” sweep configurations across 30 seeds** (instead of relying on single-run outcomes).

## What was rerun

All reruns used the project’s sweep runner:

```bash
uv run python -m src.sweep --seed-start 1 --seed-end 30 ... --aggregate
```

Rerun outputs live in `data/sweeps/` as paired files:
- `rerun_<...>.csv` (per-seed raw results)
- `rerun_<...>_agg.csv` (mean/std/min/max across the 30 seeds)

## Key takeaway

**Single-run sweep results materially overestimate performance.**

Once you look across seeds, many configs that looked “perfect” (F1=1.0) become clearly **unstable** or **overfit to one seed**, especially at very low fraud rates (e.g., 1%).

## Rerun results (aggregated)

Columns:
- `precision_mean ± precision_std`, `recall_mean ± recall_std`, `f1_mean ± f1_std`
- `flag_rate_mean` = fraction of all records flagged (mean across seeds)
- `f1_min`, `recall_min` are useful “stability floor” metrics

| file | fraud_ratio | detectors | distance_metric | fusion_strategy | threshold | eps | min_samples | if_contamination | graph_similarity_threshold | precision_mean | precision_std | recall_mean | recall_std | f1_score_mean | f1_score_std | flag_rate_mean | f1_min | recall_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rerun_200_001_db_if_graph_wavg_t07_agg.csv | 0.01 | dbscan,isolation_forest,graph | jaro_winkler | weighted_avg | 0.70 | 0.35 | 2 | auto | 0.75 | 0.929 | 0.189 | 0.867 | 0.225 | 0.862 | 0.177 | 1.023% | 0.571 | 0.500 |
| rerun_200_001_db_if_wavg_t06_agg.csv | 0.01 | dbscan,isolation_forest | jaro_winkler | weighted_avg | 0.60 | 0.40 | 3 | auto | 0.7 | 0.444 | 0.227 | 1.000 | 0.000 | 0.584 | 0.204 | 2.888% | 0.235 | 1.000 |
| rerun_400_003_db_if_wavg_t08_agg.csv | 0.03 | dbscan,isolation_forest | levenshtein | weighted_avg | 0.80 | 0.35 | 2 | 0.01 | 0.7 | 0.760 | 0.122 | 0.925 | 0.094 | 0.829 | 0.093 | 3.625% | 0.593 | 0.667 |
| rerun_500_005_db_if_wavg_t08_agg.csv | 0.05 | dbscan,isolation_forest | levenshtein | weighted_avg | 0.80 | 0.30 | 3 | 0.01 | 0.7 | 0.831 | 0.094 | 0.899 | 0.078 | 0.857 | 0.053 | 5.244% | 0.762 | 0.720 |
| rerun_500_005_db_if_graph_wavg_t06_agg.csv | 0.05 | dbscan,isolation_forest,graph | jaro_winkler | weighted_avg | 0.60 | 0.40 | 3 | 0.02 | 0.7 | 0.928 | 0.071 | 0.731 | 0.116 | 0.810 | 0.078 | 3.784% | 0.556 | 0.400 |
| rerun_300_010_db_if_graph_wavg_t07_auto_agg.csv | 0.10 | dbscan,isolation_forest,graph | levenshtein | weighted_avg | 0.70 | 0.35 | 2 | auto | 0.7 | 0.811 | 0.082 | 0.849 | 0.107 | 0.823 | 0.061 | 9.646% | 0.704 | 0.633 |
| rerun_300_010_db_if_graph_wavg_t07_c002_agg.csv | 0.10 | dbscan,isolation_forest,graph | levenshtein | weighted_avg | 0.70 | 0.40 | 2 | 0.02 | 0.65 | 0.812 | 0.083 | 0.848 | 0.109 | 0.822 | 0.061 | 9.626% | 0.704 | 0.633 |
| rerun_400_015_db_if_graph_vote_t06_agg.csv | 0.15 | dbscan,isolation_forest,graph | jaro_winkler | voting | 0.60 | 0.40 | 3 | auto | 0.7 | 0.794 | 0.045 | 0.861 | 0.072 | 0.825 | 0.050 | 14.167% | 0.691 | 0.633 |
| rerun_200_015_db_if_wavg_t07_agg.csv | 0.15 | dbscan,isolation_forest | jaro_winkler | weighted_avg | 0.70 | 0.35 | 2 | 0.01 | 0.7 | 0.783 | 0.069 | 0.861 | 0.098 | 0.814 | 0.046 | 14.507% | 0.714 | 0.633 |

## Conclusions (what seems consistently true)

1. **Weighted average fusion is the most reliable default.** Voting can work, but it’s not consistently better.
2. **Avoid low-threshold configs that “win” by flagging too much.** They can deliver high recall but unacceptable precision in practice.
3. **DBSCAN `eps≈0.35` and `min_samples≈2` are a good stable starting point.**
4. **IsolationForest contamination:** small fixed values (e.g. `0.01`) work well in mid base-rates; `auto` is workable but adds variance.
5. **Graph detector helps most at very low base rates (≈1%)** where pure anomaly/clustering can oscillate between “perfect” and “noisy” across seeds.

## Recommended default parameterization (what we should work with)

If we must pick **one default configuration to iterate on** (good balance, tested on 30-seed rerun, not overly complex):

- `detectors`: `dbscan`, `isolation_forest`
- `fusion_strategy`: `weighted_avg`
- `threshold`: `0.80`
- DBSCAN: `eps=0.35`, `min_samples=2`, `distance_metric=levenshtein`
- IsolationForest: `contamination=0.01`

This is exactly the rerun in `rerun_400_003_db_if_wavg_t08_agg.csv` (and the same family performs strongly at 5%).

### When to deviate from the default

- If you operate in **very low fraud-rate regimes (≈1–2%)** and want higher precision with low flag volume, enable `graph` and start from:
  - `graph_similarity_threshold=0.75`
  - keep `weighted_avg`, and consider `threshold≈0.70`

(That combination is reflected in `rerun_200_001_db_if_graph_wavg_t07_agg.csv`.)

## Next step (practical)

Lock the above defaults into `config/default.yaml` (or your chosen config file), then treat `threshold` as the primary “business knob” for alert volume vs. miss rate.
