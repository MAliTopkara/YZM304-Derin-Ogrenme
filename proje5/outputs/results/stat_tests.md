# Statistical Comparison

## Pairwise tests per seed

### Seed 42

| model A | model B | McNemar p | DeLong AUC_A | AUC_B | DeLong p |
|---|---|---|---|---|---|
| resnet50 | densenet121 | 0.0251 | 0.880 | 0.846 | 0.0164 |
| resnet50 | efficientnet_b0 | 0.1614 | 0.880 | 0.869 | 0.4480 |
| densenet121 | efficientnet_b0 | 0.3619 | 0.846 | 0.869 | 0.1069 |

### Seed 123

| model A | model B | McNemar p | DeLong AUC_A | AUC_B | DeLong p |
|---|---|---|---|---|---|
| resnet50 | densenet121 | 0.2286 | 0.900 | 0.853 | 0.0147 |
| resnet50 | efficientnet_b0 | 0.0718 | 0.900 | 0.856 | 0.0091 |
| densenet121 | efficientnet_b0 | 0.5611 | 0.853 | 0.856 | 0.8450 |

### Seed 2024

| model A | model B | McNemar p | DeLong AUC_A | AUC_B | DeLong p |
|---|---|---|---|---|---|
| resnet50 | densenet121 | 0.8774 | 0.893 | 0.870 | 0.0927 |
| resnet50 | efficientnet_b0 | 0.4424 | 0.893 | 0.878 | 0.3073 |
| densenet121 | efficientnet_b0 | 0.3135 | 0.870 | 0.878 | 0.6112 |

## Paired t-test on test F1 across seeds

Uses each seed as a paired observation; F1 computed at threshold 0.5.

| model A | model B | mean F1_A | mean F1_B | t | p |
|---|---|---|---|---|---|
| resnet50 | densenet121 | 0.677 | 0.651 | 1.307 | 0.3212 |
| resnet50 | efficientnet_b0 | 0.677 | 0.626 | 5.483 | 0.0317 |
| densenet121 | efficientnet_b0 | 0.651 | 0.626 | 1.226 | 0.3451 |
