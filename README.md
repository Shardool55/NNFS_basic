Ideal PyTorch Pipeline (Manual Version)
======================================

### Objective

The goal of this code is to **manually build and train a very simple neural network from scratch on the Breast Cancer dataset**, using basic PyTorch tensors and operations.  
We are **not** trying to get state-of-the-art accuracy here; the main objective is to **understand each step of the pipeline**:
- how the data is loaded and preprocessed,
- how features and labels are prepared,
- how a simple model is defined,
- how forward pass, loss, and weight updates work.

### Files

- `data_preprocessing.py` – loads the breast cancer dataset, drops unused columns, does train/test split, scaling, and label encoding.
- `model_and_training.py` – converts the NumPy arrays to PyTorch tensors, defines a tiny neural network, trains it, and evaluates accuracy.

### How to run

1. Install the required packages (for example with `pip`):
   - `numpy`, `pandas`, `scikit-learn`, `torch`
2. From this folder, run:
   - `python data_preprocessing.py`
   - `python model_and_training.py`

Side note- It is a known fact that the the code doesn't follow OOP concepts but further iterations will follow the same.

