# Optimized Vessel Segmentation: A Structure-Agnostic Approach with Small Vessel Enhancement and Morphological Correction
![Method](./method.png)

## Usage

### Installiation

```
conda create -n ovs python=3.12
conda activate ovs
git clone https://github.com/Hk416mod2/OVS-Net.git
cd /OVS-Net
pip install -r requirement.txt
```

### Data Prepare

The datase structure is as follows:

```
dataset/
├── train/
│   ├── images/
│   ├── masks/
│   └── train_val_fold.json
└── test/
    ├── images/
    ├── masks/
    └── test.json
```

We have placed some image samples and JSON generating scripts in `/Dataset`

### Train

We use SAM vit-b checkpoint for finetuning, you can download it from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

We use 5-fold(0~4) cross validation, 0 fold is used by default, you can change it in the `-fold` parameter in `train.py`

Run `python train.py`

### Test

Set checkpoint path in `python test.py`

Run `python test.py`

You can also download our model checkpoints at [here](https://drive.google.com/drive/folders/1Iq_EuKd1soDqz-aFluXi5W4vJUWgB6Tv?usp=drive_link)

## Dataset info

### Internal Validation Dataset

We use these dataset to train our model:

| Dataset            | Area                | Modality | Resolution | Split   | Access  | Related Link   |
|--------------------|---------------------|----------|------------|---------|---------|---------------------------------------------------|
| 134XCA        | Coronary Artery     | X-Ray    | 300×300    | 90/10/34  | Public |  http://personal.cimat.mx:8181/~ivan.cruz/DB_Angiograms.html|
| XCAD         | Coronary Artery     | X-Ray    | 512×512    | 75/9/42   | Public |  https://github.com/AISIGSJTU/SSVS|
| ARIA          | Retina              | Fundus   | 768×576    | 108/12/43  | Public | https://sourceforge.net/projects/aria-vessels/|
| CHASE_DB1    | Retina              | Fundus   | 960×999    | 18/2/8    | Public | https://blogs.kingston.ac.uk/retinal/chasedb1/|
| DR-HAGIS      | Retina              | Fundus   | 1024×1024  | 18/2/20   | Public | https://personalpages.manchester.ac.uk/staff/niall.p.mcloughlin/|
| DRIVE         | Retina              | Fundus   | 584×565    | 18/2/20   | Public | https://drive.grand-challenge.org/|
| FIVES         | Retina              | Fundus   | 2048×2048  | 540/60/200 | Public | https://www5.cs.fau.de/research/data/fundus-images/|
| Les-AV        | Retina              | Fundus   | 1620×1444  | 18/2/2    | Public | https://figshare.com/articles/dataset/LES-AV_dataset/11857698|
| ORVS         | Retina              | Fundus   | 1444×1444  | 38/4/7    | Public | https://github.com/AbdullahSarhan/ICPRVessels|
| STARE        | Retina              | Fundus   | 605×700    | 14/2/4    | Public | https://cecas.clemson.edu/~ahoover/stare/ |
| ROSE-1       | Retina              | OCTA     | 304×304    | 27/3/9    | Permission | https://imed.nimte.ac.cn/dataofrose.html |
| ROSSA        | Retina              | OCTA     | 320×320    | 736/82/100 | Public | https://github.com/nhjydywd/OCTA-FRNet |
| OCTA500-3M   | Retina              | OCTA     | 304×304    | 135/15/50  | Permission | https://ieee-dataport.org/open-access/octa-500 |
| OCTA500-6M   | Retina              | OCTA     | 400×400    | 180/20/100 | Permission | https://ieee-dataport.org/open-access/octa-500 |
| DrSAM        | Pelvic-Iliac Artery | X-Ray    | 386×448    | 360/40/100 | Public | https://drive.google.com/file/d/1TjxEJUD4VC_SAPcqdNVybsKRb_xW-Bze/view |


### External Validation Dataset

We use these dataset to evaluate the out-of-domain generalization of our model:

| Dataset            | Area                | Modality | Resolution | Access | Related Link   |
|--------------------|---------------------|----------|------------|---------|---------------------------------------------------|
| HRF        | Retina              | Fundus   | 3504×2336    | Public |  https://www5.cs.fau.de/research/data/fundus-images/|
| IOSTAR         | Retina              | Fundus    | 1024×1024    | Permission |  http://www.retinacheck.org/download-iostar-retinal-vessel-segmentation-dataset|
