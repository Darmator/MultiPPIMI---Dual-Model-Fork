# MultiPPIMI - Dual Model Fork
A fork of [MultiPPIMI](https://github.com/sun-heqi/MultiPPIMI/tree/main) extended with dual-task capabilities for simultaneous classification and regression prediction of protein-protein interaction modulators (PPIMIs), with additional support for drug repurposing applications.

## Overview
This repository extends the original MultiPPIMI framework to predict both binary activity labels (classification) and continuous binding affinity values (regression) for potential PPIMIs. The dual-task approach enables more nuanced bioactivity predictions and has been applied to FDA-approved drug repurposing.

<img width="1295" height="509" alt="multi_task_multippi" src="https://github.com/user-attachments/assets/ba79a633-51a3-48a9-9d37-b1efd1088bb1" />


## Key Modifications from Original Repository
### Enhanced Dual-Task Architecture
- **Dual Model Implementation**: Modified `MultiPPIMI.py` to support simultaneous classification and regression predictions
- **Ensemble Prediction**: Pre-trained ensemble of 10 neural network models for robust predictions
- **Optimized Inference**: Enhanced `PPIMI_datasets.py` with improved inference speed and support for unseen molecules

### Extended Datasets
- **FDA Drug Repurposing**: Curated dataset of FDA-approved drug structures for systematic repurposing screening
- **Expanded Protein Features**: Broader coverage of proteins with pre-processed features beyond the original repository
- **Regression Datasets**: ChEMBL-derived regression affinity labels for multiple PPI targets (`reg_data/`)

### Code reproducibility
- **`main_dual.py`**: Training script for dual-task bioactivity prediction models
- **`FDA_repurposing.ipynb`**: Jupyter notebook demonstrating batch predictions on FDA molecules with full reproducibility
- **`bioactivity_score.py`**: Simplified API for model loading and inference

## Repository Structure
```
├── data/
│   ├── FDA_structures/          # FDA-approved drugs for repurposing
│   ├── features/                # Extended protein feature set
│   │   ├── protein_esm2.csv           # ESM2 embeddings (692 proteins)
│   │   ├── protein_phy.csv            # Physicochemical properties (692 proteins)
│   │   ├── protein_binding_esm2.csv   # Additional ESM2 embeddings (6140 proteins)
│   │   └── protein_binding_phy.csv    # Additional properties (6140 proteins)
│   ├── protein_seqs.csv         # Primary protein set (692 proteins)
│   ├── protein_binding_seqs.csv # Extended protein set (6140 proteins)
│   └── reg_data/                # PPI regression affinity datasets (ChEMBL)
├── final_models/                # Pre-trained ensemble (10 NN weights)
├── src/
│   ├── MultiPPIMI.py           # Dual classification/regression model
│   ├── bioactivity_score.py    # Easy-to-use inference API
│   ├── datasets/
│   │   ├── PPIMI_datasets.py   # Enhanced dataset class with fast inference
│   └── ...
├── FDA_repurposing.ipynb        # Drug repurposing workflow & predictions
├── main_dual.py                 # Training script for dual models
├── environment.yml              # Conda environment specification
└── README.md
```

## Installation
### Option 1: Automatic Installation
Create the conda environment automatically from the provided environment file:
```bash
conda env create -f environment.yml
conda activate dual_multippimi
```

### Option 2: Manual Installation
Alternatively, create and configure the environment manually:
```bash
# Create new conda environment with Python 3.12
conda create -n dual_multippimi python=3.12
conda activate dual_multippimi

# Install PyTorch with CUDA 12.4 support
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install PyTorch Geometric
pip install torch_geometric

# Install PyTorch Geometric extensions
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

# Install additional dependencies
pip install ogb==1.3.5
pip install rdkit
```

## Usage

### Training a Dual Model
Train your own dual-task model using the modified training script:
```bash
python main_dual.py --fold 1
```

### Bioactivity Score Prediction

The `FDA_repurposing.ipynb` notebook serves as both a comprehensive tutorial for applying the model to obtain predictions and a reproducibility script for our published results.

For programmatic access, use the simplified `PPIScore` class defined in `src/bioactivity_score.py`:
```python
from src.bioactivity_score import PPIScore

# Initialize model for a specific protein-protein interaction
# You need the UniProt IDs of both proteins involved in the PPI
# Example: NCS-1 (P62166) and Ric8 (Q9NPQ8)
ric8_model = PPIScore(prot1="P62166", prot2="Q9NPQ8")  # NCS1/Ric8

# Predict bioactivity score for a small molecule (SMILES string)
score = ric8_model("OCCN(CCO)C1=NC2=C(N=C(N=C2N2CCCCC2)N(CCO)CCO)C(=N1)N1CCCCC1")
print(f"Predicted bioactivity score: {score}")

# Batch predictions on multiple molecules
smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
scores = ric8_model(smiles_list)
```

#### Available Proteins

To select a protein-protein interaction (PPI) for prediction, you need to know the **UniProt IDs** of both proteins involved in the interaction. The model currently supports predictions for **692 proteins** listed in `data/protein_seqs.csv`. An additional **6,140 proteins** are available in `data/protein_binding_seqs.csv`.

**To use proteins from the extended set** (`protein_binding_seqs.csv`):
1. Extract the protein's ESM2 embedding from `data/features/protein_binding_esm2.csv`
2. Extract the protein's physicochemical properties from `data/features/protein_binding_phy.csv`
3. Append these features to `data/features/protein_esm2.csv` and `data/features/protein_phy.csv` respectively

**For proteins not in either dataset**:
1. Generate ESM2 embeddings using the [ESM2 protein language model](https://huggingface.co/facebook/esm2_t33_650M_UR50D)
2. Calculate physicochemical properties as described in the [original MultiPPIMI paper](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01527)
3. Add the features to the appropriate CSV files following the existing format
<!--
<> ## Citation

If you use this fork in your research, please cite both the original MultiPPIMI paper and acknowledge this extension:

**Original MultiPPIMI:**
```
[Citation for original MultiPPIMI paper]
```

## Acknowledgments

This work builds upon the [MultiPPIMI](https://github.com/sun-heqi/MultiPPIMI/tree/main) framework. We thank the original authors for making their code publicly available.

## License

[Specify license - typically same as original repository]


---

For questions or issues specific to this fork, please open an issue in this repository. For questions about the base MultiPPIMI framework, refer to the [original repository](https://github.com/sun-heqi/MultiPPIMI/tree/main).
 -->
