# Self-Supervised and Invariant Wireless Channel Representations

This repository comprises of the necessary code to reproduce results in the paper [Self-Supervised and Invariant Representations for Wireless Localization](TBC) that has been submitted for review. Results, as reported in Table III, can be replicated by executing the scripts with minimal or no modifications. 

**Note:**  It is possible to achieve significantly greater accuracy than what we report in Table III for our models (See [results](./results/README.md)). A demostrated example in Colab is also provided.

### Cite
```
@JOURNAL{TBC,
    author={Salihu, Artan and Schwarz, Stefan and Rupp, Markus},
    booktitle={TBC}, 
    title={Self-Supervised and Invariant Representations for Wireless Localization}, 
    year={2023},
    pages={-},
    doi={-}
}
```

## Setup
Dependencies: 
- python (>=3.7)
- Torch: 1.10.0+cu111
- Check other imported libraries (or use Colab).
  
See provided Colab notebook.

#### Datasets
Two sample sets from the KUL-LAB-NLOS dataset are provided to test the codes and replicate the results. Alternatively, download the complete dataset `ultradense_dataset ` from [IEEE data port](https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset). To process the dataset, use the commented code in the `tester_classifier.py` script starting from line 132.

- A sample set with approximately $\mathbf{9\,000}$ samples. Download channel samples from [kuluwen_URA_lab_nLoS_CSI_9k](https://owncloud.tuwien.ac.at/index.php/s/JymbpaW4aKwP5nG/download) and labels from [kuluwen_URA_lab_nLoS_LOC_9k](https://owncloud.tuwien.ac.at/index.php/s/JymbpaW4aKwP5nG/download). Set these vales in `config.json` for the keys `sub_dataset_to_use` and `sub_loc_dataset_to_use`. Place the datasets in a subfolder and set the name of subfolder in `config.json` for the key `saved_dataset_path`.
- A sample set with approximately $\mathbf{28\,000}$ samples. Download channel samples from [kuluwen_URA_lab_nLoS_CSI_28k](https://owncloud.tuwien.ac.at/index.php/s/JymbpaW4aKwP5nG/download) and labels from [kuluwen_URA_lab_nLoS_LOC_28k](https://owncloud.tuwien.ac.at/index.php/s/JymbpaW4aKwP5nG/download). Set these vales in `config.json` for the keys `sub_dataset_to_use` and `sub_loc_dataset_to_use`. Place the datasets in a subfolder and set the name of subfolder in `config.json` for the key `saved_dataset_path`.
- Same processing steps for KUL-LOS-ULA and KUL-LOS-DIS.

#### Other datasets
- Download dataset for [S-Scenario](https://owncloud.tuwien.ac.at/index.php/s/JymbpaW4aKwP5nG/download).
- Download dataset for [HB-Scenario](https://owncloud.tuwien.ac.at/index.php/s/bkPRLaa3y3t4pjj/download).
- The **wild-v2** dataset is obtained from the 'supposed to happen' [competition](https://www.kaggle.com/competitions/wild-v2/overview).

#### SSL Training
TBC
#### Linear Location Esitmation
TBC

#### Spot Estimation
TBC

#### Fine-tuner for Location Estimation
TBC

#### Fully-Supervisor for Location Estimation
TBC

#### 

#### Acknowledgement
References [[26]](https://ieeexplore.ieee.org/document/9709990), [[30]](https://arxiv.org/pdf/2106.09785.pdf),[[31]](https://ieeexplore.ieee.org/document/9878641), and [[timm]](https://github.com/rwightman/pytorch-image-models/tree/main/timm).

### Other
TBC

#### Transfer Learning to Path-Loss Estimation
TBC