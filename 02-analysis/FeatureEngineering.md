## Run 1: Categorical baseline
Run ID: 7034b6ffa82145b988e716ad37df17a5
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.2692151359110782 ± 0.0010616857189952042

### Important features
- height
- angle 
- end_x
- duration

### Important / unimportant parameters
Generally `subsample` and `bagging_fre` are not very important parameters.

## Run 2: Scaled length + log_velocity
Scale length to SI units. Compute velocity using length and duration.

Run ID: 0d762acae19d414d8a19606dd62dd46b
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.26982492673234065 ± 0.0020796210853153657

### Important features
- height
- angle 
- end_x
- duration

### Important / unimportant parameters
`bagging_fre` and `min_child_samples` have really low importance across all runs.  `n_estimators` and `learning_rate` have some standout values 

## Run 3: Split angle into cos and sin
The ``angle`` feature has values in the range $[-\pi, \pi]$. However, according to the documentation $-\pi$ and $\pi$ represent the same direction, but value wise they are very different. We fix this by introducing two new features ``angle_sin`` and ``angle_cos`` which are just the sine and cosine values of the ``angle``. This way we can represent $-\pi$ and $\pi$ as the same angle value wise.

Run ID: ad3307b3d66f464b8729efce0b861264
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.2689565829109584 ± 0.00135282213413655

### Important features
- height
- angle_cos
- end_x
- duration

## Run 4: progressive_distance
Instead of working only with coordinates, calculate how far the pass has progressed the ball towards goal.

Run ID: 7b977190c7a64838a25eb09e7117e7db
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.2689565829109584 ± 0.00135282213413655

### Important features
- progressive_distance
- height
- end_distance_to_goal
- log_velocity

## Run 5: Direction to goal 
Make a metric for whether the pass is made towards the opposing team's goal or just sideways (e.g. cycling posssession).

Run ID: 1b311e287a65426cb3e5c7b7190bf3c9
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.2676868066954423 ± 0.0016523893384580177

### Important features
- height
- direction_to_goal_cos
- duration

### Run 6: Interaction features
Using the boolean `under_pressure`, we can create features for describing the duration, length, log_velocity when the player is under pressure.

Run ID: f6bec73525054dcf971e068c3ba832c0
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.2674971094402257 ± 0.0014154310779104493

### Important features
- height
- direction_to_goal_cos
- log_velocity

### Run 7: Group by height and then mean direction_to_goal_cos
Since height is the most important categorical feature and direction_to_goal_cos is the most important numerical, let us group by and then mean. Since this feature involves the mean operator, this needs to be applied in the inner loop columnwise.

Run ID: 900927d23b0b43f4bd62c8ccc0aa9fd8
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.26750734804977616 ± 0.0013970735807639955

Really low SHAP importance. Remove.

### Run 8: Group by height and then mean log_velocity
Same as Run 7 but mean velocity instead.

Run ID: 0db832075e1a4daeb79413078d3a2425
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.2674971094402257 ± 0.0014154310779104493

### Run 9: Group by body_part and then mean log_velocity
Run ID: 36a5dc46f7f243baa61ad1a08e5b32b3
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.2679799623308192 ± 0.0013095576881769126

### Run 10: Count number of occurences for height (categorical feature)
Run ID: 68fd329b6ebb4a6b892765849ef7d657
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.2679799623308192 ± 0.0013095576881769126