# XGBoost gunshot — PR / FN report

- PR-AUC (validation, average_precision_score): **1.000000**
- PR-AUC (test): **1.000000**
- PR curve integral (validation, trapz on precision-recall curve): **1.000000**
- `scale_pos_weight` (neg/pos on train): **1726.7143**

## Threshold (chosen on validation to minimize FN, then FP)

- **threshold** = `0.999894`

### Validation at chosen threshold

{
  "threshold": 0.999893678894043,
  "tn": 2591,
  "fp": 0,
  "fn": 0,
  "tp": 1,
  "rule": "min_positive_score_minus_epsilon"
}

### Test set confusion matrix (tn, fp, fn, tp)

{
  "tn": 2590,
  "fp": 0,
  "fn": 1,
  "tp": 1
}
