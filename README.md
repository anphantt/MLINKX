# M-LINKX: Multiview Graph Learning for Brain Cognitive Disease Detection

<!-- Badges -->
[![Paper](https://img.shields.io/badge/Paper-IEEE-B31B1B.svg)](https://arxiv.org/abs/2608.14847)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-CAUEEG%20%7C%20AHEAP-green.svg)](#datasets)

---

## 📖 Abstract

Electroencephalogram (EEG) is a non-invasive and relatively low-cost procedure that measures brain electricity for the detection of cognitive diseases. EEG-based classification of dementia-related conditions, including Alzheimer's disease (AD), mild cognitive impairment (MCI), and frontotemporal dementia (FTD), remains challenging because EEG signals are noisy, non-stationary, and vary across subjects. Segment-based learning provides a practical way to model long EEG recordings by converting them into fixed-length inputs. 

In this paper, we propose **M-LINKX**, a multi-view graph learning framework for EEG-based dementia classification. For each segment, we extract channel-level node features and construct multiple functional-connectivity (FC) graph views, where each view is defined by a specific combination of connectivity metric, frequency band, and topology filter[cite: 3]. Instead of relying on traditional message passing, M-LINKX separately models node features and adjacency-based connectivity representations inspired by LINKX. Graph-view representations are fused using global trainable view weights, and subject-level predictions are obtained via soft voting across segment-level probabilities. Extensive experiments on two 3-class EEG datasets (CAUEEG and AHEAP) show that M-LINKX achieves state-of-the-art subject-level performance.

---

## ✨ Key Features

- **Multi-View Functional Connectivity (FC) Graphs:** Captures complex channel-interaction patterns across different metrics (Coherence, wPLI), frequency bands ($\delta, \theta, \alpha, \beta, \gamma$), and graph topology filters (Complete, Domain, Top-k, Hybrid).
- **Separate Encoding Architecture (LINKX-style):** Avoids direct feature mixing over noisy FC connections by independently encoding channel node attributes and adjacency graph views.
- **Adaptive View Weight Fusion:** Automatically learns global trainable weights to assign dynamic importance to complementary connectivity graph views].
- **Subject-Level Soft Voting:** Integrates segment-level probability distributions to form robust, subject-level clinical diagnostic predictions.

---

## Method

![M-LINKX Framework](/home/anphan/Documents/graph/mlinkx/image/mlinkx.png)

*Figure 1: Overview of the proposed M-LINKX framework[cite: 6]. (1) Subject-level EEG recordings are segmented into fixed-length windows[cite: 6]. (2) Feature extraction and multi-view FC graph construction[cite: 6]. (3) M-LINKX encoder separately encodes node features and adjacency views, applies trainable view weight fusion, and aggregates segment probabilities for final subject classification[cite: 6].*

---

## 📂 Repository Structure

```text
MLINKX/
├── data/                   # Dataset directory (CAUEEG, AHEAP)
│   ├── caueeg/             # Raw / Preprocessed CAUEEG files
│   └── aheap/              # Raw / Preprocessed AHEAP files
├── src/                    # Source code directory
│   ├── datasets/           # Data loaders and dataset preprocessing
│   ├── feature_extraction/ # Bandpower, spectral entropy, and FC graph view construction
│   ├── models/             # Architecture implementations (M-LINKX, LINKX, GATv2)
│   └── utils/              # Helper functions (metrics, logging, visualization)
├── configs/                # Configuration files (hyperparameters, views config)
│   ├── caueeg_config.yaml
│   └── aheap_config.yaml
├── preprocess.py           # Data preprocessing and feature extraction script
├── train.py                # Main training script (10-fold CV)
├── evaluate.py             # Inference and evaluation script (Subject-level soft voting)
├── requirements.txt        # Python package dependencies
├── LICENSE                 # License file
└── README.md               # Project documentation

```

---

## 🛠️ Environment Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/anphantt/MLINKX.git](https://github.com/anphantt/MLINKX.git)
cd MLINKX

```

### 2. Create and Activate Conda Environment

```bash
conda create -n mlinkx python=3.10 -y
conda activate mlinkx

```

### 3. Install Dependencies

Install PyTorch according to your system's CUDA version, then install the remaining requirements:

```bash
# Example for PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install project dependencies
pip install -r requirements.txt

```

---

## 📊 Datasets

We evaluate M-LINKX on two public 3-class EEG datasets for dementia classification:

1. **CAUEEG Dataset:**
* **Classes:** Healthy Controls (HC), Mild Cognitive Impairment (MCI), Alzheimer's Disease (AD).


* **Channels:** 19 EEG channels (10-20 international system).


* **Segment Window:** Default 10 seconds.

* *Note:* The CAUEEG dataset is not redistributed in this repository. To request access to the full CAUEEG dataset, please follow the instructions provided by the dataset authors: (https://github.com/ipis-mjkim/caueeg-dataset)


2. **AHEAP Dataset:**
* **Classes:** Healthy Controls (HC), Alzheimer's Disease (AD), Frontotemporal Dementia (FTD).


* **Channels:** 19 EEG channels.


* **Segment Window:** Default 4 seconds.





### Data Preparation & Preprocessing

Place raw EEG data into the `data/` folder following this structure:

```text
data/
├── caueeg/
│   ├── subjects_info.csv
│   └── raw_eeg/
└── aheap/
    ├── subjects_info.csv
    └── raw_eeg/

```

To extract node features (bandpower, spectral entropy) and construct multi-view FC graphs, run:

```bash
python preprocess.py --dataset caueeg --window_length 10
python preprocess.py --dataset aheap --window_length 4

```

---

## 🚀 Usage

### 1. Training M-LINKX

To train M-LINKX using 10-fold subject-based cross-validation on CAUEEG:

```bash
python train.py --config configs/caueeg_config.yaml --device cuda:0

```

To train on AHEAP:

```bash
python train.py --config configs/aheap_config.yaml --device cuda:0

```

### 2. Subject-Level Evaluation

To run inference and compute subject-level classification metrics via Soft Voting aggregation:

```bash
python evaluate.py --checkpoint checkpoints/mlinkx_caueeg_best.pt --dataset caueeg

```

---

## 🔬 Experimental Results

### Subject-Level Performance Comparison

Subject-level classification performance on **CAUEEG** (10-second window) and **AHEAP** (4-second window) under 3-class classification settings:

| Model | Raw EEG | Node Features | FC Input | CAUEEG (Balanced Acc.) | CAUEEG (Macro-F1) | AHEAP (Balanced Acc.) | AHEAP (Macro-F1) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **1D-ResNet**<br> | $\checkmark$ |  |  | 0.6146 ± 0.0135

 | 0.6224 ± 0.0141

 | 0.5221 ± 0.0432

 | 0.4815 ± 0.0511

 |
| **CNN-LSTM**<br> | $\checkmark$ |  |  | 0.5585 ± 0.0049

 | 0.5700 ± 0.0133

 | 0.5697 ± 0.0527

 | 0.5494 ± 0.0438

 |
| **MLP-node**<br> |  | $\checkmark$ |  | 0.5949 ± 0.0233

 | 0.5961 ± 0.0199

 | 0.5550 ± 0.0446

 | 0.5740 ± 0.0477

 |
| **Ensemble**<br> | $\checkmark$ | $\checkmark$ |  | 0.6048 ± 0.0303

 | 0.6084 ± 0.0285

 | 0.5791 ± 0.0258

 | 0.5528 ± 0.0273

 |
| **CNN**<br> |  |  | $\mathcal{V}$<br> | 0.5100 ± 0.0241

 | 0.5210 ± 0.0243

 | 0.4596 ± 0.0426

 | 0.4758 ± 0.0312

 |
| **LINKX**<br> |  |  | $v^*$<br> | 0.6154 ± 0.0179

 | 0.6077 ± 0.0198

 | 0.5806 ± 0.0384

 | 0.6060 ± 0.0460

 |
| **GATv2**<br> |  |  | $v^*$<br> | 0.5868 ± 0.0190

 | 0.5910 ± 0.0236

 | 0.4901 ± 0.0393

 | 0.5383 ± 0.0255

 |
| **Multi-view GATv2**<br> |  |  | $\mathcal{V}$<br> | 0.5916 ± 0.0320

 | 0.5992 ± 0.0316

 | 0.5895 ± 0.0187

 | 0.6041 ± 0.0303

 |
| **M-LINKX (Ours)**<br> |  |  | $\mathcal{V}$<br> | **0.6665 ± 0.0389**<br> | **0.6649 ± 0.0294**<br> | **0.6191 ± 0.0281**<br> | **0.5990 ± 0.0257**<br> |

### Visualizations

<p align="center">
  <img src="/home/anphan/Documents/graph/mlinkx/image/new_CF.png" width="80%" />
  <br>
  <em>Figure 2: Subject-level confusion matrices for CAUEEG and AHEAP datasets[cite: 8].</em>
</p>
---

## 📝 Citation

If you use this codebase or method in your research, please consider citing our paper:

```bibtex
@inproceedings{phan2026mlinkx,
  title={M-LINKX: Multiview Graph Learning for Brain Cognitive Disease Detection},
  author={Phan, An and Jin, Yufei and Zhu, Xingquan},
  booktitle={25th IEEE International Conference on Machine Learning and Applications (ICMLA)},
  year={2026}
}

```
