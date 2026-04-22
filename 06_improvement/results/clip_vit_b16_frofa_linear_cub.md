# CLIP frozen feature + FroFA linear probe results

| Method | Backbone | Classifier | Setting | Accuracy | 95% CI |
|---|---|---|---|---:|---:|
| no-FroFA | CLIP ViT-B/16 | closed-form L2 linear probe | 5-way 1-shot | 86.975% | +/- 0.569% |
| FroFA | CLIP ViT-B/16 | closed-form L2 linear probe | 5-way 1-shot | 85.717% | +/- 0.574% |
| no-FroFA | CLIP ViT-B/16 | closed-form L2 linear probe | 5-way 5-shot | 96.379% | +/- 0.248% |
| FroFA | CLIP ViT-B/16 | closed-form L2 linear probe | 5-way 5-shot | 95.733% | +/- 0.295% |
