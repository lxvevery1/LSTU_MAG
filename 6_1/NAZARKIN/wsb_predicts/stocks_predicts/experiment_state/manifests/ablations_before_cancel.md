# Seeds and ablations

Generated from **14** evaluated runs.

## Seed runs (full model)

| run | n | macro-F1 | MCC | AUC | ECE | ticker-day F1 | ticker-day MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| finbert_h5_seed42 | 1270 | 0.4302 | 0.1559 | 0.6016 | 0.0665 | 0.4024 | 0.1284 |
| finbert_h5_seed43 | 1270 | 0.4124 | 0.1387 | 0.6007 | 0.0762 | 0.4045 | 0.1403 |
| finbert_h5_seed44 | 1270 | 0.4399 | 0.1587 | 0.6145 | 0.0590 | 0.4334 | 0.1474 |

Mean ± std over available seeds:

| metric | mean | std | seeds |
|---|---:|---:|---:|
| macro_f1 | 0.4275 | 0.0140 | 3 |
| mcc | 0.1511 | 0.0109 | 3 |
| accuracy | 0.4315 | 0.0052 | 3 |
| auc | 0.6056 | 0.0077 | 3 |
| auc_pr_up | 0.3810 | 0.0253 | 3 |
| brier | 0.5993 | 0.0041 | 3 |
| ece | 0.0672 | 0.0086 | 3 |
| ticker_day_macro_f1 | 0.4134 | 0.0173 | 3 |
| ticker_day_mcc | 0.1387 | 0.0096 | 3 |
| ticker_day_auc | 0.5954 | 0.0145 | 3 |

## Ablations (seed 42)

| ablation | macro-F1 | Δ macro-F1 | MCC | Δ MCC | AUC | ECE | Δ ticker-day F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ablate_frozen_encoder_seed42 | 0.3948 | -0.0354 | 0.1837 | +0.0278 | 0.6123 | 0.0523 | -0.0330 |
| ablate_no_aux_loss_seed42 | 0.3437 | -0.0865 | 0.2739 | +0.1180 | 0.6192 | 0.0153 | -0.0822 |
| ablate_no_class_probs_seed42 | 0.4086 | -0.0216 | 0.1088 | -0.0471 | 0.5927 | 0.0812 | -0.0215 |
| ablate_no_influence_weight_seed42 | 0.3830 | -0.0471 | 0.1741 | +0.0182 | 0.6018 | 0.0492 | -0.0283 |
| ablate_no_noise_meme_seed42 | 0.4160 | -0.0142 | 0.1177 | -0.0382 | 0.6012 | 0.0934 | -0.0069 |
| ablate_no_ticker_fragment_seed42 | 0.4099 | -0.0202 | 0.1512 | -0.0047 | 0.5993 | 0.0644 | -0.0258 |
| ablate_single_horizon_h5_seed42 | 0.4235 | -0.0066 | 0.1419 | -0.0140 | 0.5989 | 0.0782 | -0.0161 |
| ablate_text_only_seed42 | 0.4184 | -0.0118 | 0.1420 | -0.0139 | 0.6129 | 0.0579 | -0.0001 |
| ablate_weighted_ce_only_seed42 | 0.4302 | +0.0000 | 0.1559 | +0.0000 | 0.6016 | 0.0665 | +0.0000 |

Reference: `finbert_h5_seed42` macro-F1 0.4302, MCC 0.1559, AUC 0.6016, ECE 0.0665.

Negative deltas mean the ablated model is worse on the same test set. Ablations that change the test population (`no_noise_meme`) are not directly paired with the reference and are marked by their row count.

## Encoder / max-length comparisons

| run | n | macro-F1 | MCC | AUC | ECE |
|---|---:|---:|---:|---:|---:|
| finbert_h5_maxlen512_seed42 | 1270 | 0.4087 | 0.1561 | 0.6037 | 0.0522 |
| minilm_frozen_h5_seed42 | 1270 | 0.4048 | 0.1286 | 0.6008 | 0.0616 |
