# TGDAnet: Text-Guided Domain Adaptation  Network for Hyperspectral Binary Change Detection


> **TGDAnet** is a novel deep learning framework that integrates **text-guided semantics** with **unsupervised domain adaptation (UDA)** for cross-domain hyperspectral image change detection (HSI-CD).

<p align="center">
  <img src="Flowchart of the proposed TGDAnet.png" alt="TGDAnet Flowchart" width="90%">
</p>

---

## 📖 Overview

Change detection in hyperspectral images is a fundamental task in remote sensing. However, models trained on one domain (source) often fail to generalize to another domain (target) due to domain shift. **TGDAnet** addresses this challenge by:

- **Domain Adaptation** — We adopt a domain adaptation  strategy that enables effective knowledge transfer from the SD to TD to address the challenge of limited labeled data in HSIs-CD.
- **Vision-Text Model** — We construct both the generalized  and specialized text features for SD and TD, respectively. By fusing vision and text modalities, the model benefits from enhanced semantic guidance, resulting in more generalized and robust cross-domain CD features. 
- **Contrastive Learning** — We introduce global and local  contrastive loss functions to better capture subtle and discriminative features. This design improves the sensitivity of the model to fine-grained variations across domains, thereby enhancing the overall effectiveness of CD.
---

## 📊 Datasets

Six hyperspectral datasets are supported for cross-domain change detection:

| Dataset | Location | Time 1 | Time 2 | Image Size | Bands |
|---------|----------|--------|--------|------------|-------|
| **China** | Yancheng, Jiangsu, China | 2006-05-03 | 2007-04-23 | 450×140 | 155 |
| **USA** | USA | 2004-05-01 | 2007-05-08 | — | — |
| **SantaBarbara** | California, USA | 2013 | 2014 | 984×740 | 224 |
| **BayArea** | California, USA | 2013 | 2015 | 600×500 | 224 |

Each dataset contains bi-temporal `.mat` files and a binary ground truth map (0 = no change, 1 = change, 2 = background).

---

## 🚀 Quick Start

### Requirements

```bash
pip install torch torchvision numpy scipy scikit-learn matplotlib einops tqdm
```

- Python ≥ 3.8
- PyTorch ≥ 1.7
- CUDA (recommended)

### Project Structure

```
TGDAnet_CD/
├── Code/
│   ├── TGDAnet.py                 # Main model definition
│   ├── Transformer.py             # Spectral-Spatial Transformer
│   ├── Text_Encode_Module.py      # Text encoding & alignment module
│   ├── InfoNCE.py                 # Contrastive loss & domain adaptation loss
│   ├── clip.py                    # CLIP model loader & tokenizer
│   ├── simple_tokenizer.py        # BPE tokenizer (CLIP-style)
│   ├── datasets_utils.py          # Data loading, patching, batch generation
│   ├── GeneratePic.py             # Visualization utilities
│   ├── Demo_Bay2Santa.py          # Train: BayArea → SantaBarbara
│   ├── Demo_Santa2Bay.py          # Train: SantaBarbara → BayArea
│   ├── Demo_China2USA.py          # Train: China → USA
│   ├── Demo_USA2China.py          # Train: USA → China
│   ├── ViT-B-32.pt                # CLIP ViT-B/32 pretrained weights
│   ├── bpe_simple_vocab_16e6.txt.gz # BPE vocabulary
│   └── result/                    # Output predictions (.mat + .png)
├── Data/
│   ├── 01_China/                  # China dataset (.mat + .tif)
│   ├── 03_USA/                    # USA dataset
│   ├── 05_SantaBarbara/           # Santa Barbara dataset
│   └── 06_BayArea/                # Bay Area dataset
├── Flowchart of the proposed TGDAnet.png
└── README.md
```

### Training

Run one of the demo scripts to train a cross-domain change detection model:

```bash
# Example: Transfer from BayArea (source) → SantaBarbara (target)
python Code/Demo_Bay2Santa.py --cuda 0 --epoches 50 --batches 256 --lr_rate 1e-4 --lambda1 1e-2 --lambda2 1e-2
```

Available transfer tasks:
- `Demo_Bay2Santa.py` — BayArea → SantaBarbara
- `Demo_Santa2Bay.py` — SantaBarbara → BayArea
- `Demo_China2USA.py` — China → USA
- `Demo_USA2China.py` — USA → China

### Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--epoches` | Training epochs | 50 |
| `--batches` | Batch size | 256 |
| `--patches` | Patch size | 11 |
| `--lr_rate` | Learning rate | 1e-4 |
| `--lambda1` | Weight for domain adaptation loss | 1e-2 |
| `--lambda2` | Weight for text alignment loss | 1e-2 |
| `--tr_rate` | Training sample ratio (%) | 2.5 |
| `--val_rate` | Validation sample ratio (%) | 10.0 |

---

## 📈 Evaluation Metrics

The model reports the following metrics on the target domain:

- **IoU** (Intersection over Union / Jaccard Index)
- **F1 Score**
- **OA** (Overall Accuracy)
- **Kappa** (Cohen's Kappa)
- **Precision**
- **Recall**

### Sample Results

| Task | IoU | F1 | OA | Kappa | Precision | Recall|
|------|-----|----|----|-------|-------|-------|
| Bay → Santa | 84.53% | 91.62% | 93.16% | 0.8586 | 0.8846 | 0.9501 |
| Santa → Bay | 90.52% | 95.03% | 94.79% | 0.8957 | 0.9710 | 0.9304 |
| China → USA | 76.38% | 86.61% | 94.09% | 0.8281 | 0.8845 | 0.8484 |
| USA → China | 86.25% | 92.62% | 95.54% | 0.8943 | 0.8901 | 0.9653 |

---

## 📝 Citation

If you use TGDAnet in your research, please cite:

```bibtex
@article{qin2026tgdanet,
  title={TGDAnet: Text-Guided Domain Adaptation Network for Hyperspectral Binary Change Detection},
  author={Qin, Xuexiang and Zhang, Yuxiang and Xu, Mingming and Gan, Wenxia and Dong, Yanni},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026},
  publisher={IEEE}
}
```

---

## 📄 License

This project is for research purposes only. Please contact the authors for usage permissions.

---

## 🙏 Acknowledgments
- The remote sensing datasets from various public sources
