# MultiPPIMI - Dual Model Fork

A fork of [MultiPPIMI](https://github.com/sun-heqi/MultiPPIMI/tree/main) extended with dual-task capabilities for simultaneous classification and regression prediction of protein-protein interaction modulators (PPIMIs), with additional support for drug repurposing applications.

## Overview

This repository extends the original MultiPPIMI framework to predict both binary activity labels (classification) and continuous binding affinity values (regression) for potential PPIMIs. The dual-task approach enables more nuanced bioactivity predictions and has been applied to FDA-approved drug repurposing.

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

## Repository Structure

```
├── data/
│   ├── FDA_structures/          # FDA-approved drugs for repurposing
│   ├── features/                # Extended protein feature set
│   └── reg_data/                # PPI regression affinity datasets (ChEMBL)
├── final_models/                # Pre-trained ensemble (10 NN weights)
├── src/
│   ├── MultiPPIMI.py           # Dual classification/regression model
|   ├── datasets/
|   │   ├── PPIMI_datasets.py       # Enhanced dataset class with fast inference
│   └── ...
├── FDA_repurposing.ipynb        # Drug repurposing workflow & predictions
├── main_dual.py                 # Training script for dual models
└── README.md
```

## Installation

Please follow the installation instructions from the [original MultiPPIMI repository](https://github.com/sun-heqi/MultiPPIMI/tree/main).


### Training a Dual Model

Train your own dual-task model using the modified training script:

```bash
python main_dual.py --fold 1
```

### Drug Repurposing Workflow

The `FDA_repurposing.ipynb` notebook provides a complete pipeline for:
- Loading FDA-approved drug structures
- Running ensemble predictions
- Ensuring full reproducibility of predictions

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
