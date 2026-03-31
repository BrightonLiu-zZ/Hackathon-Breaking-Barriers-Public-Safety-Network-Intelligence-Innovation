# XGBoost gunshot — PR / FN report

- PR-AUC (validation, average_precision_score): **0.827629**
- PR-AUC (test): **0.849057**
- PR curve integral (validation, trapz on precision-recall curve): **0.911140**
- `scale_pos_weight` (neg/pos on train): **1726.9923**

## Threshold (chosen on validation to minimize FN, then FP)

- **threshold** = `0.741499`

### Validation at chosen threshold

{
  "threshold": 0.7414989471435547,
  "tn": 138144,
  "fp": 16,
  "fn": 2,
  "tp": 78,
  "rule": "f2_score_maximisation"
}

### Test set confusion matrix (tn, fp, fn, tp)

{
  "tn": 155415,
  "fp": 16,
  "fn": 0,
  "tp": 90
}
