# Stage 2: flagged vs. unflagged pairs (M5 control)

Same metrics as the submitted RQ2 (anchor_shift, delta_top_rel); flagged = label mismatch + semantic drift; unflagged = pairs passing all three checks. Refusal mismatches excluded (single image).


## GPT-Image

refusal mismatch, generated_side=source: n=96, anchor in top-30% = 56.2%  
refusal mismatch, generated_side=followup: n=70, anchor in top-30% = 61.4%  
| population | n | MR-introduced in more salient half | anchor more salient | anchor less salient | mean anchor_shift |
|---|---|---|---|---|---|
| flagged (label+drift) | 206 | 81.6% | 51.5% | 45.1% | +0.013 |
| unflagged | 1058 | 75.8% | 47.2% | 46.2% | +0.023 |
| level2_mismatch | 113 | 73.5% | 52.2% | 44.2% | +0.007 |
| level3_drift | 93 | 91.4% | 50.5% | 46.2% | +0.020 |

MR-introduced-in-salient-half: flagged − unflagged = +5.7 pp, 95% CI [+0.3, +11.5], permutation p = 0.0838  
mean anchor_shift: flagged − unflagged = -0.010, permutation p = 0.688

## Gemini-Image

refusal mismatch, generated_side=source: n=112, anchor in top-30% = 54.5%  
refusal mismatch, generated_side=followup: n=27, anchor in top-30% = 44.4%  
| population | n | MR-introduced in more salient half | anchor more salient | anchor less salient | mean anchor_shift |
|---|---|---|---|---|---|
| flagged (label+drift) | 246 | 85.0% | 49.2% | 45.9% | +0.035 |
| unflagged | 1203 | 72.2% | 45.3% | 46.6% | +0.006 |
| level2_mismatch | 77 | 74.0% | 48.1% | 45.5% | +0.109 |
| level3_drift | 169 | 89.9% | 49.7% | 46.2% | +0.002 |

MR-introduced-in-salient-half: flagged − unflagged = +12.7 pp, 95% CI [+7.6, +17.5], permutation p = 0.0001  
mean anchor_shift: flagged − unflagged = +0.029, permutation p = 0.2198

## SDXL

| population | n | MR-introduced in more salient half | anchor more salient | anchor less salient | mean anchor_shift |
|---|---|---|---|---|---|
| flagged (label+drift) | 596 | 78.0% | 44.1% | 48.8% | +0.043 |
| unflagged | 1390 | 73.2% | 46.7% | 46.3% | +0.018 |
| level2_mismatch | 337 | 76.0% | 49.6% | 41.5% | -0.009 |
| level3_drift | 259 | 80.7% | 37.1% | 58.3% | +0.112 |

MR-introduced-in-salient-half: flagged − unflagged = +4.8 pp, 95% CI [+0.9, +9.0], permutation p = 0.0288  
mean anchor_shift: flagged − unflagged = +0.025, permutation p = 0.1211
