# UniVesselSeg

![Method](./method.png)

## Installiation

- Create a new conda environment
```
conda create -n univesselseg python=3.10
conda activate univesselseg
```

- Install UniVesselSeg
```
git clone https://github.com/Hk416mod2/UniVesselSeg
cd UniVesselSeg/
pip install -r requirement.txt
```
## Train

### Data Prepare
The datase structure is as follows:
```
dataset/
├── train/
│   ├── images/
│   ├── masks/
│   ├── image2label_train.json
│   └── image2label_val.json
└── test/
    ├── images/
    ├── masks/
    └── image2label_test.json
```
We have prepared a script for generating JSON in `dataset`

### Train model
`python train.py`

### Test
`python test.py`

you can download our model checkpoints at [here](https://drive.google.com/drive/folders/1cF5BMRBkTyZNYsUxDOR2EgE-g8Kq-hg8?usp=sharing)

