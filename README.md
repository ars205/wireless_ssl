# Self-Supervised and Invariant Wireless Channel Representations

The implementation and models to reproduce main results of [Self-Supervised and Invariant Representations for Wireless Localization](https://arxiv.org/pdf/2302.07000.pdf).

### System Model
An illustration of the system model for the co-located massive MIMO scenarios is shown below.

<p align="center">
  <img src="assets/System_model_SSL2022_v4.jpg" width="350" height="200"/>
</p>


## Setup
Dependencies: 
- python (>=3.7)
- Torch: 1.10.0+cu111
- Check other imported libraries (or use Colab).

The results from Table III in the paper can be replicated with minimal or no modifications to the provided scripts.

**Note:**  It is possible to achieve significantly greater accuracy than what we report in Table III for our models (See [results](./results/README.md)). See Colab notebook for a [demonstration](https://drive.google.com/file/d/1BHFtmGYVj2VWEXTKWjAVp502mIyuBiTr/view?usp=sharing).

## Models
The weights for different trained models can be accessed by downloading from the table below. The example provided contains the weights for the model trained on the KUL-NLOS dataset, and with the complete set of channel transformations.

| Model | Download | Dataset |
|------------------------|-------------------------------|----|
| model_nlos_10800 |  [Download from owncloud](https://owncloud.tuwien.ac.at/index.php/s/3dvqqD2WZkA5bSV/download) | KUL-URA-NLOS | 
| model_nlos_05400 |  [Download from owncloud](https://owncloud.tuwien.ac.at/index.php/s/bQZ1HrZmBisfaEa/download) | KUL-URA-NLOS |
| model_nlos_63k |  [Download](https://drive.google.com/file/d/1GWxpevuL9susuOXHSgp51wRSPigeNA64/view?usp=drive_link) | KUL-URA-NLOS |
| model_nlos_9k_07500 |  [Download from owncloud](https://owncloud.tuwien.ac.at/index.php/s/bQZ1HrZmBisfaEa/download) | KUL-URA-NLOS (improved transferability for spot estimation) |
| model_los_dis_01080 |  [Download](https://drive.google.com/uc?export=download&id=10Gpk_ulw1e7Gelh1CG2Frx5WczOnLkYv) | KUL-DIS-LOS |
| model_los_dis_01800 |  [Download](https://drive.google.com/uc?export=download&id=1YryIaq_p1CPU43wDyFqHRl2GIkKtQyK) | KUL-DIS-LOS |
| model_los_dis_10800 |  [Download](https://drive.google.com/uc?export=download&id=1gnt7FPq_k5ijbNghbf9fSKIBQR2BGGKB) | KUL-DIS-LOS |
| model_los_ula_01000 |  [Download](https://drive.google.com/uc?export=download&id=1Nd4yzH4PObbFSGUkNG9rrHjo5kc37NoQ) | KUL-ULA-LOS |
| model_los_ula_02100 |  [Download](https://drive.google.com/uc?export=download&id=1sT_0651s_vE8MCI3s5vM7-4xb2d9Cr7p) | KUL-ULA-LOS |
| model_los_ula_10800 |  [Download](https://drive.google.com/uc?export=download&id=1aqFpltgvVhNsTweBEO9Uc9rvlDZXhDDd) | KUL-ULA-LOS |
| model_s_200_w_aug_R12k_01 |  [Download](https://drive.google.com/uc?export=download&id=1sYCJmdIUaMsKGFK0DLajpHy9X0ukvyET) | S-200 |
| model_s_200_wo_aug_R12k_02 |  [Download](https://drive.google.com/uc?export=download&id=1Icc70wZ1PMcTtQdZesf1ybU9zCxfgI8l) | S-200 |
| model_s_200_w_aug_28k_03 |  [Download](https://drive.google.com/uc?export=download&id=1t2ZJbJCDj0LdZlPK5OmJxtdg0RJ1Poe2) | S-200 |
| model_hb_200_w_aug_28k_01 |  [Download](https://drive.google.com/uc?export=download&id=13vlZHRqAL3QwN4HhmnmgfpjITXXBt2jb) | HB-200 |
| model_hb_200_w_aug_28k_02 |  [Download](https://drive.google.com/uc?export=download&id=1KwM-Y_37pk8bti6rXxSjoB9f92bBNlyd) | HB-200 |
#### Datasets
Two sample sets from the KUL-LAB-NLOS dataset are provided to test the codes and replicate the results. Alternatively, download the complete dataset `ultradense_dataset ` from [IEEE data port](https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset). To process the dataset, use the commented code in the `tester_classifier.py` script starting from line 132.
- A sample set with approximately $\mathbf{9\,000}$ samples. Download channel samples from [kuluwen_URA_lab_nLoS_CSI_9k](https://owncloud.tuwien.ac.at/index.php/s/JymbpaW4aKwP5nG/download) and labels from [kuluwen_URA_lab_nLoS_LOC_9k](https://owncloud.tuwien.ac.at/index.php/s/JymbpaW4aKwP5nG/download). Set these vales in `config.json` for the keys `sub_dataset_to_use` and `sub_loc_dataset_to_use`. Place the datasets in a subfolder and set the name of subfolder in `config.json` for the key `saved_dataset_path`.
- A sample set with approximately $\mathbf{28\,000}$ samples. Download channel samples from [kuluwen_URA_lab_nLoS_CSI_28k](https://owncloud.tuwien.ac.at/index.php/s/JymbpaW4aKwP5nG/download) and labels from [kuluwen_URA_lab_nLoS_LOC_28k](https://owncloud.tuwien.ac.at/index.php/s/JymbpaW4aKwP5nG/download). Set these vales in `config.json` for the keys `sub_dataset_to_use` and `sub_loc_dataset_to_use`. Place the datasets in a subfolder and set the name of subfolder in `config.json` for the key `saved_dataset_path`.
- Same processing steps for KUL-LOS-ULA and KUL-LOS-DIS. 
  - A sample set for KUL-LOS-ULA can be obtained from [kuluwen_ULA_lab_LoS_CSI_28k](https://drive.google.com/uc?export=download&id=13MA6z-hKGyG-8E44Nb8mAU65uI2mylZl) for channel estimates and [kuluwen_ULA_lab_LoS_LOC_28k](https://drive.google.com/uc?export=download&id=1vVMdi7NCWjhfEMdctVUuFeSguNV8PzvP) for labels.
  -   A sample set for KUL-LOS-DIS can be obtained from [kuluwen_DIS_lab_LoS_CSI_28k](https://drive.google.com/uc?export=download&id=1Oa51cLGUSDwKr03ja4N22kHjNyzhD2Bq) for channel estimates and [kuluwen_DIS_lab_LoS_LOC_28k](https://drive.google.com/uc?export=download&id=1Rjmb4pDr9-QokjlhtJ4SX-b6mT7mkdnR) for labels.

<p align="center"><img src="assets/KUL_Scenarios.jpg" class="center" width="500" height="200"></p>

#### S and HB datasets
Two different :train2: scenarios are modelled for the S and HB datasets. 

<p align="center"><img src="assets/S_and_HB_scenarios_2.jpg" class="center" width="700" height="300"></p>

- Download dataset for [S-Scenario](https://owncloud.tuwien.ac.at/index.php/s/JymbpaW4aKwP5nG/download). S-200 dataset details [⏭️](https://mcg-deep-wrt.netlify.app/deep-wrt/s-scenario/).
  - A sample set for S-200 can be obtained from [S_200_CSI_28k](https://drive.google.com/uc?export=download&id=1iJETQ-jS5LwbGlqp-5iw2NjdofnkHPtL) for channel estimates and [S_200_LOC_28k](https://drive.google.com/uc?export=download&id=1casM95sY7b63HuindepSutUMiyYRsGjX) for labels.
  - Similarly, for path-loss labels, a small set from [S_200_CSI_PL_28k](https://drive.google.com/uc?export=download&id=1fyHA8_IWvcX_eAqIgZGenS-ivBeuL744) for channel estimates and [S_200_LOC_PL_28k](https://drive.google.com/uc?export=download&id=1YBnvBELaAnt4BKGT5jES7zAFkeY0_Oaj) for labels.
- Download dataset for [HB-Scenario](https://owncloud.tuwien.ac.at/index.php/s/bkPRLaa3y3t4pjj/download). HB-200 dataset details [⏭️](https://mcg-deep-wrt.netlify.app/deep-wrt/hb-scenario/).
  - A sample set for HB-200 can be obtained from [HB_200_CSI_28k](https://drive.google.com/uc?export=download&id=1F6tDDNbosd-suisxHZTEuABqB-6L4uZf) for channel estimates and [HB_200_LOC_28k](https://drive.google.com/uc?export=download&id=1hPmukMQYZkD3jZU5Wwd40Ep0U3PpDjqC) for labels.

#### Other datasets
- The **wild-v2** dataset is obtained from the 'supposed to happen' [competition](https://www.kaggle.com/competitions/wild-v2/overview).

#### Linear Location Esitmation 
| Iterations_model (data regime) | ↓ MAE [mm] | ↓ 95-th percentile [mm] |
|--------------------------------|------------|-------------------------|
| model_nlos_05400 (1k) |  461.039 | 916.478 | 
| model_nlos_05400 (5k) |  402.631 | 810.78 | 
| model_nlos_05400 (10k) |  395.424 | 791.735 |

#### Spot Estimation
| Model (data regime)           | ↑ Top-1 |
|-------------------------------|-------|
| Random (2.3k)                 | 24.98 |
| model_nlos_05400 (2.3k) | 96.84 |

#### Fine-tuner for Location Estimation

| Iterations_model (data regime) | ↓ MAE [mm] | ↓ 95-th percentile [mm] |
|--------------------------------|------------|-------------------------|
| model_nlos_05400 (1k) |  280.531 | 640.626 |
| model_nlos_05400 (5k) |  130.325 | 309.206 |
| model_nlos_05400 (10k) |  81.8767 | 182.098 |

#### Fully-Supervisor for Location Estimation
| Iterations_model (data regime) | ↓ MAE [mm] | ↓ 95-th percentile [mm] |
|--------------------------------|-------------|------------------------|
| model_nlos_05400 (1k) |  345.489 | 747.552 |
| model_nlos_05400 (5k) |  158.137 | 373.993 |
| model_nlos_05400 (10k) |  92.8073 | 210.09 |

#### Transfer Learning to Path-Loss Estimation
| Method  | ↓ MAE      | ↓ 95-th           |
| ------- | ----------|------------------ |
| Fully-Supervised             | 5.917     | 18.493           |
| Transfer-learning (Linear)  | 16.08     | 31.682           |
| SWiT+Linear                  | 6.594     | 18.426           |

#### Transfer Learning to Other Environments
- Simply choose the weights from the list of trained models, and apply it to other datasets. Any of the models should achieve the same *spot estimation* accuracy as in the trained dataset.
- To better observe the inability of random weights, select a large sample size (e.g., use the 28k dataset and keep $R_{\text{test}} = 5000$). For the line-of-sight cases, the $\verb|top-1|$ accuracy should be $<5\%$.


#### Acknowledgement
References: [[27]](https://ieeexplore.ieee.org/document/9709990), [[31]](https://arxiv.org/pdf/2106.09785.pdf), [[32]](https://ieeexplore.ieee.org/document/9878641), and [[timm]](https://github.com/rwightman/pytorch-image-models/tree/main/timm).


### Cite
```
@misc{salihu2023selfsupervised,
    title={Self-Supervised and Invariant Representations for Wireless Localization},
    author={Artan Salihu and Stefan Schwarz and Markus Rupp},
    year={2023},
    eprint={2302.07000},
    archivePrefix={arXiv},
    primaryClass={eess.SP}
}
```

#### Other Works


|               Work       |   |  
|---------------------------------|---|
| [RRH Selection](https://ieeexplore.ieee.org/document/9815773) | <p align="center"><img src="assets/RRH_Selection.gif" class="center" width="150" height="120"></p> |
| [Uncertainty Wireless Loc](https://ieeexplore.ieee.org/document/9616218)        | <p align="center"><img src="assets/error_detection.gif" class="center" width="150" height="120"></p> |
| Channel Subspace                | <p align="center"><img src="assets/channel_subspace.gif" class="center" width="150" height="120"></p>|
| [Low-dimensional Representations](https://ieeexplore.ieee.org/document/9253408)        | <p align="center"><img src="assets/low_dim_rep_loc.jpg" class="center" width="150" height="120"></p>  |
