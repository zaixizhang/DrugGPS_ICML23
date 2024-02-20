# Learning Subpocket Prototypes for Generalizable Structure-based Drug Design
Official Pytorch implementation of ICML'23 paper "Learning Subpocket Prototypes for Generalizable Structure-based Drug Design"
(https://arxiv.org/abs/2305.13997). 
<div align=center><img src="https://github.com/zaixizhang/DrugGPS_ICML23/blob/main/druggps.png" width="700"/></div>

## Previous work
- DrugGPS is based on our previous work on structure-based drug design: *Molecule Generation For Target Protein Binding With Structural Motifs* (ICLR 2023)
  - Code: https://github.com/zaixizhang/FLAG
  - Paper: https://openreview.net/forum?id=Rq13idF0F73


## Install conda environment via conda yaml file
```bash
conda env create -f druggps_env.yaml
conda activate druggps_env
```

## Datasets
Please refer to [`README.md`](./data/README.md) in the `data` folder.

## Preprocess dataset with BINANA
```
cd utils
python preprocess_data.py
```

## Training

```
python train.py
```

## Sampling

```
python sample.py
```

## Generated Molecules for CrossDocked dataset
The generated molecules for 100 protein targets are stored in [druggps.pt](https://github.com/zaixizhang/DrugGPS_ICML23/blob/8853927dbccca74b5de45048635578ca47a64676/druggps.pt)

The index file is [test_index.pkl](https://github.com/zaixizhang/DrugGPS_ICML23/blob/8853927dbccca74b5de45048635578ca47a64676/test_index.pkl)

## Reference
```
@article{zhang2023learning,
  title={Learning Subpocket Prototypes for Generalizable Structure-based Drug Design},
  author={Zhang, Zaixi and Liu, Qi},
  journal={ICML},
  year={2023}
}
```
