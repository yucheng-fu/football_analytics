# Model Iterations and Feature Engineering

## Run 1 — Categorical Baseline

### Summary

| Metric | Value |
|---|---|
| Run ID | `7034b6ffa82145b988e716ad37df17a5` |
| Outer folds | 3 |
| 95% CI | `0.2692 ± 0.0011` |

### Important Features

- `height`
- `angle`
- `end_x`
- `duration`

### Hyperparameter Observations

The parameters below showed relatively low importance across tuning runs:

- `subsample`
- `bagging_freq`

---

## Run 2 — Scaled Length + Log Velocity

Scale pass length to SI units and compute velocity using length and duration.

### Summary

| Metric | Value |
|---|---|
| Run ID | `0d762acae19d414d8a19606dd62dd46b` |
| Outer folds | 3 |
| 95% CI | `0.2698 ± 0.0021` |

### Important Features

- `height`
- `angle`
- `end_x`
- `duration`

### Hyperparameter Observations

Low-importance parameters:

- `bagging_freq`
- `min_child_samples`

Parameters with stronger influence:

- `n_estimators`
- `learning_rate`

---

## Run 3 — Trigonometric Angle Representation

The `angle` feature has values in the range:

$$
[-\pi, \pi]
$$

According to the documentation, $-\pi$ and $\pi$ represent the same direction, despite having numerically different values. To address this discontinuity, two new features were introduced:

- `angle_sin`
- `angle_cos`

These encode the angle using its sine and cosine values, ensuring equivalent directions have similar representations.

### Summary

| Metric | Value |
|---|---|
| Run ID | `ad3307b3d66f464b8729efce0b861264` |
| Outer folds | 3 |
| 95% CI | `0.2690 ± 0.0014` |

### Important Features

- `height`
- `angle_cos`
- `end_x`
- `duration`

---

## Run 4 — Progressive Distance

Instead of relying solely on coordinates, calculate how far the pass progresses the ball toward goal.

### Summary

| Metric | Value |
|---|---|
| Run ID | `7b977190c7a64838a25eb09e7117e7db` |
| Outer folds | 3 |
| 95% CI | `0.2690 ± 0.0014` |

### Important Features

- `progressive_distance`
- `height`
- `end_distance_to_goal`
- `log_velocity`

---

## Run 5 — Direction to Goal

Introduce a metric describing whether a pass is directed toward the opponent’s goal or played sideways/backward during possession circulation.

### Summary

| Metric | Value |
|---|---|
| Run ID | `1b311e287a65426cb3e5c7b7190bf3c9` |
| Outer folds | 3 |
| 95% CI | `0.2677 ± 0.0017` |

### Important Features

- `height`
- `direction_to_goal_cos`
- `duration`

---

## Run 6 — Interaction Features

Using the boolean `under_pressure`, interaction features were created to describe:

- `duration`
- `length`
- `log_velocity`

under pressure situations.

### Summary

| Metric | Value |
|---|---|
| Run ID | `f6bec73525054dcf971e068c3ba832c0` |
| Outer folds | 3 |
| 95% CI | `0.2675 ± 0.0014` |

### Important Features

- `height`
- `direction_to_goal_cos`
- `log_velocity`

---

## Run 7 — Grouped Mean Direction-to-Goal by Height

Since `height` was the most important categorical feature and `direction_to_goal_cos` the most important numerical feature, a grouped mean feature was introduced.

The transformation groups by `height` and computes the mean `direction_to_goal_cos`. Because this operation uses aggregation, it must be computed column-wise within the inner CV loop to avoid leakage.

### Summary

| Metric | Value |
|---|---|
| Run ID | `900927d23b0b43f4bd62c8ccc0aa9fd8` |
| Outer folds | 3 |
| 95% CI | `0.2675 ± 0.0014` |

### Observation

Very low SHAP importance. Feature removed in later runs.

---

## Run 8 — Grouped Mean Log Velocity by Height

Same setup as Run 7, but using grouped mean `log_velocity` instead.

### Summary

| Metric | Value |
|---|---|
| Run ID | `0db832075e1a4daeb79413078d3a2425` |
| Outer folds | 3 |
| 95% CI | `0.2675 ± 0.0014` |

---

## Run 9 — Grouped Mean Log Velocity by Body Part

Group by `body_part` and compute mean `log_velocity`.

### Summary

| Metric | Value |
|---|---|
| Run ID | `36a5dc46f7f243baa61ad1a08e5b32b3` |
| Outer folds | 3 |
| 95% CI | `0.2680 ± 0.0013` |

---

## Run 10 — Height Frequency Encoding

Count the number of occurrences for the categorical feature `height`.

### Summary

| Metric | Value |
|---|---|
| Run ID | `68fd329b6ebb4a6b892765849ef7d657` |
| Outer folds | 3 |
| 95% CI | `0.2680 ± 0.0013` |

---

## Run 11 — OFE

### Summary

| Metric | Value |
|---|---|
| Run ID | `8e9a2395894d4a0a8ad20e7d198199d8` |
| Outer folds | 3 |
| 95% CI | `0.2534 ± 0.0028` |

### Observation

This configuration produced the best overall generalisation estimate across all experiments.