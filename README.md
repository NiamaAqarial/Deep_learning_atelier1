# Atelier 1: Machine Learning with PyTorch - Regression and Classification

## Overview
This repository contains the code and results for Atelier 1, focusing on two main tasks using PyTorch on GPU:
- **Part 1: Regression** - Predicting the closing stock prices from historical data (using `prices-split-adjusted.csv` or `prices.csv`).
- **Part 2: Multi-Class Classification** - Predictive maintenance to classify failure types in machinery (using `predictive_maintenance.csv`).

The code is split into two scripts:
- `atelier1_gpu.py`: Handles Part 1 (Regression).
- `atelier1_2.py`: Handles Part 2 (Classification).

Key libraries used: Pandas, NumPy, Matplotlib, Seaborn, PyTorch, Scikit-learn.

Environment setup:
- PyTorch version: 2.6.0+cu124
- CUDA available: True (Version: 12.4)
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (Memory: 8.59 GB)
- Device used: cuda

Datasets:
- Stock prices: ~851,264 rows, 7 columns (date, symbol, open, close, low, high, volume).
- Predictive maintenance: 10,000 rows, 10 columns (UDI, Product ID, Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Target, Failure Type).

## Part 1: Regression – Predicting Closing Stock Prices
### Description
This part focuses on regressing the `close` price using features: `open`, `high`, `low`, `volume`. Data is split (80/20), standardized, and trained using a feedforward neural network with ReLU activations and dropout. Hyperparameters are tuned via manual grid search. The model is trained on GPU with Adam optimizer and MSE loss.

### EDA Summary
- Apple stock price shows upward trend from 2010-2017.
- Close prices are right-skewed.
- High correlation between price variables (open, high, low, close ~1.0), negative with volume.
- Trading volume decreases over years.

### Model Architecture
- Input: 4 features.
- Hidden layers: Configurable (e.g., [256, 128, 64]).
- Output: 1 (close price).
- Activation: ReLU.
- Dropout: Configurable.
- Batch size: 256.
- Epochs: 50 (grid search), 150 (final).

### Grid Search (In Progress)
Grid parameters:
- Layers: [[128,64,32], [256,128,64], [64,32]]
- Learning rates: [0.001, 0.0005, 0.01]
- Dropout: [0.2, 0.3, 0.4]

Current results (execution ongoing, partial output):
- {'dropout': 0.2, 'layers': [128, 64, 32], 'lr': 0.001} → Test MSE = 54.90
- {'dropout': 0.2, 'layers': [128, 64, 32], 'lr': 0.0005} → Test MSE = 49.03
- {'dropout': 0.2, 'layers': [128, 64, 32], 'lr': 0.01} → Test MSE = 364.18
- {'dropout': 0.2, 'layers': [256, 128, 64], 'lr': 0.001} → Test MSE = 12.05
- {'dropout': 0.2, 'layers': [256, 128, 64], 'lr': 0.0005} → Test MSE = 7.98
- {'dropout': 0.2, 'layers': [256, 128, 64], 'lr': 0.01} → Test MSE = 169.15
- {'dropout': 0.2, 'layers': [64, 32], 'lr': 0.001} → Test MSE = 47.39
- {'dropout': 0.2, 'layers': [64, 32], 'lr': 0.0005} → Test MSE = 22.97
- {'dropout': 0.2, 'layers': [64, 32], 'lr': 0.01} → Test MSE = 92.84
- {'dropout': 0.3, 'layers': [128, 64, 32], 'lr': 0.001} → Test MSE = 37.99
- {'dropout': 0.3, 'layers': [128, 64, 32], 'lr': 0.0005} → Test MSE = 32.20
- {'dropout': 0.3, 'layers': [128, 64, 32], 'lr': 0.01} → Test MSE = 204.40
- {'dropout': 0.3, 'layers': [256, 128, 64], 'lr': 0.001} → Test MSE = 50.11
- {'dropout': 0.3, 'layers': [256, 128, 64], 'lr': 0.0005} → Test MSE = 9.85
- {'dropout': 0.3, 'layers': [256, 128, 64], 'lr': 0.01} → Test MSE = 229.63
- {'dropout': 0.3, 'layers': [64, 32], 'lr': 0.001} → Test MSE = 91.08
- {'dropout': 0.3, 'layers': [64, 32], 'lr': 0.0005} → Test MSE = 36.28

### Interpretation (Based on Current Results)
- Deeper networks ([256,128,64]) perform better, with lowest MSE ~7.98 (dropout=0.2, lr=0.0005).
- Higher LR (0.01) leads to worse performance (higher MSE), suggesting instability.
- Dropout 0.2 seems slightly better than 0.3 so far.
- Shallower models ([64,32]) have higher MSE, indicating need for more capacity.
- Execution is ongoing; final best model and training curves will be updated upon completion.

### Figures
![Figure 1: EDA for Stock Prices](<img width="1536" height="802" alt="Figure_1" src="https://github.com/user-attachments/assets/bfe4763e-c7f3-49e2-ae0b-cbf5cc896496" />

)  
*(Evolution du prix Apple, Distribution des prix de clôture, Corrélation des variables prix, Volume moyen par année)*

Note: Final training loss curves, RMSE, and R² for the best model are pending completion.

## Part 2: Multi-Class Classification – Predictive Maintenance
### Description
This part classifies failure types (6 classes) using features: Type, Air/Process temperature, Rotational speed, Torque, Tool wear. Data is imbalanced, so oversampled to balance classes. Split (80/20), standardized, and trained on GPU with Adam optimizer and CrossEntropy loss.

### EDA Summary
- Highly imbalanced: "No Failure" dominates (9652/10000).
- Correlations: Torque and Rotational speed negatively correlated (-0.88); Tool wear weakly correlated.

### Data Preparation
- Dropped irrelevant columns (UDI, Product ID).
- Encoded 'Type' (L=0, M=1, H=2) and 'Failure Type' (LabelEncoder).
- Oversampled minorities to match majority (9652 each, total ~57,912 samples).
- Stratified split.

### Model Architecture
- Input: 6 features.
- Hidden layers: [256 (BN+ReLU+Dropout), 128 (BN+ReLU+Dropout), 64 (ReLU)].
- Output: 6 classes.
- Batch size: 128.
- Epochs: 60 (grid search), 120 (final).

### Grid Search
Grid parameters:
- Learning rates: [0.001, 0.0005]
- Dropout: [0.3, 0.5]

Results:
- lr=0.001 dropout=0.3 → Accuracy = 0.9908
- lr=0.0005 dropout=0.3 → Accuracy = 0.9893
- lr=0.001 dropout=0.5 → Accuracy = 0.9820
- lr=0.0005 dropout=0.5 → Accuracy = 0.9807

Best: {'dropout': 0.3, 'lr': 0.001} | Accuracy = 0.9908

### Final Training
- Optimizer: Adam (lr=0.001, weight_decay=1e-5).
- Epochs: 120.
- Progress:
  - Epoch 20: Train Acc=0.9845 | Test Acc=0.9921
  - Epoch 40: Train Acc=0.9863 | Test Acc=0.9934
  - Epoch 60: Train Acc=0.9865 | Test Acc=0.9934
  - Epoch 80: Train Acc=0.9870 | Test Acc=0.9928
  - Epoch 100: Train Acc=0.9867 | Test Acc=0.9945
  - Epoch 120: Train Acc=0.9879 | Test Acc=0.9921

### Results
- Final Test Accuracy: 0.9921
- Classification Report:

| Class                  | Precision | Recall | F1-Score | Support |
|------------------------|-----------|--------|----------|---------|
| Heat Dissipation Failure | 0.9943 | 1.0000 | 0.9972 | 1930 |
| No Failure             | 1.0000 | 0.9528 | 0.9759 | 1930 |
| Overstrain Failure     | 0.9979 | 1.0000 | 0.9990 | 1931 |
| Power Failure          | 0.9959 | 1.0000 | 0.9979 | 1931 |
| Random Failures        | 0.9913 | 1.0000 | 0.9956 | 1931 |
| Tool Wear Failure      | 0.9743 | 1.0000 | 0.9870 | 1930 |
| **Accuracy**           |           |        | 0.9921 | 11583 |
| **Macro Avg**          | 0.9923 | 0.9921 | 0.9921 | 11583 |
| **Weighted Avg**       | 0.9923 | 0.9921 | 0.9921 | 11583 |

### Interpretation
- Excellent performance post-oversampling (Test Acc >99%).
- "No Failure" has lower recall (0.95), possibly due to original imbalance.
- Loss decreases steadily; accuracy plateaus around 0.99.
- Model generalizes well, with minimal overfitting (train/test gaps small).
- Oversampling and batch norm helped handle imbalance and training stability.

### Figures
![Figure 2: EDA for Predictive Maintenance](<img width="1400" height="500" alt="Figure_2" src="https://github.com/user-attachments/assets/3e3b95e4-51da-4210-b7fe-fabd9c0fb681" />

)  
*(Distribution des types de panne, Corrélation des variables numériques)*

![Figure 3: Training Curves for Classification](<img width="1400" height="500" alt="Figure_2_2" src="https://github.com/user-attachments/assets/d86d0068-3a0e-4345-a31e-6df43d1d605a" />

)  
*(Loss/Epochs - Classification, Accuracy/Epochs - Classification)*

## How to Run
1. Ensure GPU setup and libraries installed.
2. Run `atelier1_gpu.py` for Part 1.
3. Run `atelier1_2.py` for Part 2.
4. Datasets must be in the same directory.

## Notes
- Part 1 execution is ongoing; update README with final results.
- All plots saved during execution (e.g., via plt.show()).
- Training complete for Part 2 on cuda device.
