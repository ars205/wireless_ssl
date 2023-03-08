# For the classifier:

See results in [classfier vs random](KUL_NLOS_R28k_B256_E100/classifier_rand_vs_models_w_aug.png) or [table](KUL_NLOS_R28k_B256_E100/knn_results.txt).


# For the regressor:
Results are summarized below. A more comprehensive set of results, including the ECDF curves, are in [results for this example](KUL_NLOS_R28k_B256_E100).


## Linear Head


**Results for Linear Regressor KUL-NLOS (Complete set of augmentations used during the SSL) (linear-head trained for 501 epochs).**


| Iterations_model (data regime) | Transformation during testing | Experiment name (model) | MAE (L1Loss) | 95-th percentile | RMSE (MSELoss) |
|------------------------|-------------------------------|--------------------------|-------------|-----------------|---------------|
| model_nlos_10800 (1k) | ('CenterSubcarriers()', 'NormalizeMaxValue()') | KUL_NLOS_w_aug_R28k_B256_E100_v2 | 449.55 | 913.876 | 498.206 |
| model_nlos_10800 (5k) | ('CenterSubcarriers()', 'NormalizeMaxValue()') | KUL_NLOS_w_aug_R28k_B256_E100_v2 | 413.083 | 821.78 | 456.582 |
| model_nlos_10800 (10k) | ('CenterSubcarriers()', 'NormalizeMaxValue()') | KUL_NLOS_w_aug_R28k_B256_E100_v2 | 405.83 | 800.334 | 449.221 |
-------------------------------------------------------------------------------------------------------------------------------------------------
| Iterations_model (data regime) | Transformation during testing | Experiment name (model) | MAE (L1Loss) | 95-th percentile | RMSE (MSELoss) |
|------------------------|-------------------------------|--------------------------|-------------|-----------------|---------------|
| model_nlos_05400 (1k) | ('CenterSubcarriers()', 'NormalizeMaxValue()') | KUL_NLOS_w_aug_R28k_B256_E100_v2 | 461.039 | 916.478 | 509.394 |
| model_nlos_05400 (5k) | ('CenterSubcarriers()', 'NormalizeMaxValue()') | KUL_NLOS_w_aug_R28k_B256_E100_v2 | 402.631 | 810.78 | 446.343 |
| model_nlos_05400 (10k) | ('CenterSubcarriers()', 'NormalizeMaxValue()') | KUL_NLOS_w_aug_R28k_B256_E100_v2 | 395.424 | 791.735 | 437.035 |



## Linear Head and fine-tuning


**Results for Fine Tuner KUL-NLOS (Complete set of augmentations used during the training) (fine-tuned **only** for 151 epochs).**


| Iterations_model (data regime) | Transformation during testing                 | Experiment name (model) | MAE (L1Loss) | 95-th percentile | RMSE (MSELoss) |
| ---------------------- | -------------------------------------------- | ----------------------- | ------------ | ---------------- | -------------- |
| model_nlos_10800 (1k) | ('CenterSubcarriers()', 'NormalizeMaxValue()') | KUL_NLOS_w_aug_R28k_B256_E100_v2 | 295.134 | 678.368 | 323.675 |
| model_nlos_10800 (5k) | ('CenterSubcarriers()', 'NormalizeMaxValue()') | KUL_NLOS_w_aug_R28k_B256_E100_v2 | 134.088 | 308.003 | 148.895 |
| model_nlos_10800 (10k)| ('CenterSubcarriers()', 'NormalizeMaxValue()') | KUL_NLOS_w_aug_R28k_B256_E100_v2 | 73.5547 | 170.92 | 81.8537 |


| Iterations_model (data regime) | Transformation during testing                 | Experiment name (model) | MAE (L1Loss) | 95-th percentile | RMSE (MSELoss) |
| ---------------------- | -------------------------------------------- | ----------------------- | ------------ | ---------------- | -------------- |
| model_nlos_05400 (1k) | ('CenterSubcarriers()', 'NormalizeMaxValue()') | KUL_NLOS_w_aug_R28k_B256_E100_v2 | 280.531 | 640.626 | 308.205 |
| model_nlos_05400 (5k) | ('CenterSubcarriers()', 'NormalizeMaxValue()') | KUL_NLOS_w_aug_R28k_B256_E100_v2 | 130.325 | 309.206 | 144.706 |
| model_nlos_05400 (10k)| ('CenterSubcarriers()', 'NormalizeMaxValue()') | KUL_NLOS_w_aug_R28k_B256_E100_v2 | 81.8767 | 182.098 | 91.0124 |



## Random weights


**Results for Fully Supervisor KUL-NLOS (Random weights) for 151 epochs.**


| Iterations_model (data regime) | Transformation during testing | Experiment name (model) | MAE (L1Loss) | 95-th percentile | RMSE (MSELoss) |
| --- | --- | --- | --- | --- | --- |
| Random (1k) | ('CenterSubcarriers()', 'NormalizeMaxValue()') | KUL_NLOS_w_aug_R28k_B256_E100_v2	 | 345.489 | 747.552 | 382.703 |
| Random (5k) | ('CenterSubcarriers()', 'NormalizeMaxValue()') | KUL_NLOS_w_aug_R28k_B256_E100_v2	 | 158.137 | 373.993 | 175.243 |
| Random (10k) | ('CenterSubcarriers()', 'NormalizeMaxValue()') | KUL_NLOS_w_aug_R28k_B256_E100_v2	 | 92.8073 | 210.09 | 102.914 |


