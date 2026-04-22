# Stage 6 final summary

| Experiment | Method | 5-way 1-shot | 5-way 5-shot | Main conclusion |
|---|---|---:|---:|---|
| Global CLIP feature | no-FroFA linear probe | 86.975 +/- 0.569 | 96.379 +/- 0.248 | Strong frozen feature baseline |
| Global CLIP feature | FroFA linear probe | 85.717 +/- 0.574 | 95.733 +/- 0.295 | Pooled embedding FroFA hurts |
| Patch-token paired | MAP | 46.213 +/- 0.875 | 75.478 +/- 0.765 | Patch-token MAP baseline |
| Patch-token paired | FroFA + MAP | 45.402 +/- 0.890 | 77.056 +/- 0.819 | FroFA improves 5-shot by +1.578 |

Final interpretation: CLIP frozen features strongly improve over task-trained Conv64F/ResNet12 baselines. FroFA should not be applied directly to pooled global embeddings. When moved to ViT patch tokens and paired-episode MAP-head evaluation, FroFA shows a clear 5-shot gain while 1-shot still needs tuning.
