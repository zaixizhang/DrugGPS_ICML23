# Learning Subpocket Prototypes for Generalizable Structure-based Drug Design
Official Pytorch implementation of ICML'23 paper "Learning Subpocket Prototypes for Generalizable Structure-based Drug Design"
(https://arxiv.org/abs/2305.13997). 
<div align=center><img src="https://github.com/zaixizhang/DrugGPS_ICML23/blob/main/druggps.png" width="700"/></div>


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

## Reference
```
@article{zhang2023learning,
  title={Learning Subpocket Prototypes for Generalizable Structure-based Drug Design},
  author={Zhang, Zaixi and Liu, Qi},
  journal={ICML},
  year={2023}
}
```
