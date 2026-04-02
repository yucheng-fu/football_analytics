## Run 1: Baseline

Run ID: 958972dfb7e94bd29268d8bb462bd3be
95% confidence interval for best estimate of generalisation: 0.2765114817047379 ± 0.014952746941420321

## Run 2: Scaled length + log_velocity
Run ID: f1b80a25b9994d9e82540a94b824d304
95% confidence interval for best estimate of generalisation: 0.2690960604637107 ± 0.0038091663547390056

Verdict: KEEP

## Run 3.1: Split angle into cos and sin
Run ID: ce451d3139f947539ba078c21020713a
95% confidence interval for best estimate of generalisation: 0.26840982210642217 ± 0.002504415208136691

## Run 3.2: Keep only angle_cos
Run ID: f341658bc8fc4545945f2c4099e61ad2
95% confidence interval for best estimate of generalisation: 0.27043317546039036 ± 0.0007109422917909562

VERDICT: Keep both sin and cos of angle

## Run 4: progressive_distance  
Run ID: 766b9bb3857a4c93b0e0622fa9f821f6
95% confidence interval for best estimate of generalisation: 0.2697785621161741 ± 0.0030402601711667027

VERDICT: Slightly lower loss but very good feature

## Run 4.1: start and end distance_to_goal
Run ID: 
95% confidence interval for best estimate of generalisation: 0.2662614024261866 ± 0.004596739229966727

## Run 4.2 Drop start_distance_to_goal
Run ID: 
95% confidence interval for best estimate of generalisation: 0.2662614024261866 ± 0.004596739229966727

## Run 4.3 Remove start x and start y
Run ID:
95% confidence interval for best estimate of generalisation: 0.2678540351972088 ± 0.002046960307232844

## Run 5: 
Run ID: 
95% confidence interval for best estimate of generalisation: 0.26758208652253557 ± 0.0028403758698786453




# NEW

## Run 1: OHE baseline
Run ID: 22c686d079964456b2f0162e7ab00c4b
Number of outer folds: 5
95% confidence interval for best estimate of generalisation: 0.2725873919631917 ± 0.007469400513935643

### Important features
- height_Ground Pass
- angle
- end_x
- duration

### Important / unimportant parameters
No clear consensus

## Run 1.1: Categorical baseline
Run ID: 8f617821926e4ac0a8cfbac335fe9fbd
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.2686750934180138 ± 0.0025723875542576313

### Important features
- height
- angle 
- end_x
- duration

### Important / unimportant parameters
Generally `subsample` and `bagging_fre` are not very important parameters.

## Run 2: Scaled length + log_velocity + categorical
Run ID: 3b43a17a6bea4d6bbca9d514ee7bff7a
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.27175046282118803 ± 0.00377022730500416  

### Important features
- height
- angle 
- end_x
- duration

The log_velocity feature did not have more importance than these four, but it seems to be a slightly better feature than just length alone.

### Important / unimportant parameters
`bagging_fre` and `min_child_samples` have really low importance across all runs.  `n_estimators` and `learning_rate` have some standout values 

# Run 2.1: Scaled length + log_velocity + ohe
Run ID: a005eb50df214f3c9f228be0003c2637
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.27114668577227546 ± 0.00020294702082168756

### Important features
- height_Ground Pass
- angle 
- end_x
- duration

### Important / unimportant parameters
`min_child_samples` have really low importance across all runs.  `n_estimators`, `num_leaves` and `reg_lambda` have some standout values 

## Run 3: Split angle into cos and sin
Run ID: 89815248785044c8930689dd192e571d
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.27060438890624117 ± 0.005860787388809485

### Important features
- height_Ground Pass
- angle_cos
- end_x
- duration

## Run 3.1: only keep cos
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.2747245680862579 ± 0.008216162868337824

Don't do that.

## Run 4: progressive_distance
Run ID: 63e7e76529ab41db918c422f2a9716fd
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.2674235984730591 ± 0.0010315111035873637

### Important features
- progressive_distance
- height_Ground Pass
- end_x

## Run 4.1: add end_distance_to_goal
Run ID: ddb744ff353c4089b06066d3b90210b5
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.2691957122228088 ± 0.0049096023209388735

### Important features
- progressive_distance
- Height_Ground pass
- angle_cos

end_distance_to_goal is better than end_x

## Run 5: add dx dy
Run ID: cb8ea9b4730545ccab50c84d5ba7287b
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.2678837484508101 ± 0.004065049186665466

### Important features
- progressive_distance
- Height_Ground pass
- end_distance to goal

dx and dy do not really add anything. Removing. 

## Run 6: remove dx dy and use categorical
Run ID: a5138fcc35c34d5db9d255140787395a
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.2666992264131723 ± 0.0018212117693168692

### Important features
- progressive_distance
- height
- angle

## Run 7: under_pressure as int
Run ID: 79b9feaf6e1e4423ac1817b1f2d034fc
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.2694259865371965 ± 0.0027416771422592405

Let's keep it for now. 

## Run 8: Direction to goal 
Run ID: 
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.2662358138498291 ± 0.000612552762747068

### Important features
- height
- direction_to_goal_cos
- duration

## Run 9: Pressure length and angle
Run ID: a1be94a6875a4b7f95dbc65f89dced70
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.27022432100318533 ± 0.00526449264463523

yeah, don't use these features.

## Run 10: Remove top 7 bottom features
Run ID: 5254a3c6122b41379ba7661ef27f993a
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.27692532700894906 ± 0.009766143507010308

Horrible idea, don't do this!!

## Run 11: run 8 + OFE
Run ID: 849066b56654405fa97b2d9e33448d78
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.25536887399704555 ± 0.008869908667940936

Pretty good.

## Run 12: Run 8 + removing no features
Run ID: f0c13283392c40d5a00cddf1cb95b267
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.26618754398904043 ± 0.0022417939202031343

## Run 13: same as 12 but with OHE encoding
363ae53267e84aff9638b3b18726bec6
Number of outer folds: 3
95% confidence interval for best estimate of generalisation: 0.2711627880054668 ± 0.004163082240990599