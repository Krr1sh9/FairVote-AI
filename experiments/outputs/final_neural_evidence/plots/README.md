# Final neural evidence plots

These plots are generated from `experiments/outputs/final_neural_evidence/`.

Important: this is a computationally constrained evidence run, not the exhaustive full preset.
It used one trial per condition, five MRP training steps, and five neural training steps.
Use the plots to illustrate the generated evidence, not as proof that the neural model is
universally superior.

## Plots

1. `overall_l1_by_epsilon_method.png`  
   Mean overall L1 error by Randomized Response epsilon and estimator method. Lower is better.

2. `worst_group_l1_by_epsilon_method.png`  
   Mean worst-group L1 error by epsilon and estimator method. Lower is better.  
   Worst-group L1 is computed as the larger of region and age worst-group L1 for each condition.

3. `overall_l1_by_sample_size_method.png`  
   Mean overall L1 error by sample size and estimator method. Lower is better.

4. `winner_correctness_by_method.png`  
   Mean winner correctness rate by method. Higher is better.

5. `runtime_by_method_log_scale.png`  
   Mean runtime by method, shown on a log scale. Lower is better.

6. `neural_vs_linear_overall_l1_delta_by_epsilon.png`  
   Difference in overall L1 between neural RR-MRP and linear RR-aware MRP by epsilon.  
   Negative values favour neural; positive values favour linear.

## Recommended use

Main dissertation:
- `overall_l1_by_epsilon_method.png`
- `neural_vs_linear_overall_l1_delta_by_epsilon.png`
- optionally `winner_correctness_by_method.png`

Appendix:
- `worst_group_l1_by_epsilon_method.png`
- `overall_l1_by_sample_size_method.png`
- `runtime_by_method_log_scale.png`
