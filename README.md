# Learning Subpocket Prototypes for Generalizable Structure-based Drug Design
Official Pytorch implementation of ICML'23 paper "Learning Subpocket Prototypes for Generalizable Structure-based Drug Design"
(https://arxiv.org/abs/2110.00987). 
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

