# NLP Final Project - Analyzing Dataset Artifacts in SNLI

CS388 Natural Language Processing Final Project

## Project Overview

Analyzing and mitigating dataset artifacts in the Stanford Natural Language Inference (SNLI) dataset using ELECTRA-small-discriminator.

### Goals
1. Train baseline model on SNLI
2. Identify dataset artifacts through hypothesis-only baseline
3. Implement debiasing method (Dataset Cartography)
4. Evaluate improvements
## Project Structure

```
├── notebooks/          # Jupyter notebooks for experiments
│   ├── 01_baseline_training.ipynb
│   ├── 02_hypothesis_only.ipynb
│   ├── 03_analysis.ipynb
│   └── 04_improved_model.ipynb
├── scripts/           # Python scripts
│   ├── train_baseline.py
│   ├── hypothesis_only_baseline.py
│   ├── analysis_starter.py
│   └── dataset_cartography.py
├── outputs/           # Model checkpoints (saved to Google Drive, not in git)
├── analysis/          # Analysis results and plots
│   ├── plots/
│   └── results/
├── paper/            # LaTeX paper
└── requirements.txt  # Dependencies
```

## Setup

### On Google Colab
```python
!git clone https://github.com/sonalshreya/NLP_project.git
%cd NLP_project
!pip install -r requirements.txt
```

### Local Setup
```bash
git clone https://github.com/sonalshreya/NLP_project.git
cd NLP_project
pip install -r requirements.txt
```

## Usage

### Training Baseline Model
```bash
python scripts/train_baseline.py
```

### Running Analysis
```bash
python scripts/analysis_starter.py
```

