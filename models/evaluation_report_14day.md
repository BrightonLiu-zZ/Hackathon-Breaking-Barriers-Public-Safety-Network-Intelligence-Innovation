# XGBoost gunshot — PR / FN report

- PR-AUC (validation, average_precision_score): **1.000000**
- PR-AUC (test): **0.666667**
- PR curve integral (validation, trapz on precision-recall curve): **1.000000**
- `scale_pos_weight` (neg/pos on train): **17269.7000**

## Threshold (chosen on validation to minimize FN, then FP)

- **threshold** = `0.524978`

### Validation at chosen threshold

{
  "threshold": 0.5249780416488647,
  "tn": 34540,
  "fp": 0,
  "fn": 0,
  "tp": 2,
  "rule": "f2_score_maximisation"
}

### Test set confusion matrix (tn, fp, fn, tp)

{
  "tn": 34540,
  "fp": 1,
  "fn": 0,
  "tp": 2
}
