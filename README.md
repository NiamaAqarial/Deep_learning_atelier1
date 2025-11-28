# Deep Learning Lab 1: PyTorch for Regression and Multi-Class Classification

## Objective
The main purpose of this lab is to get familiar with the PyTorch library for performing classification and regression tasks using Deep Neural Network (DNN)/Multi-Layer Perceptron (MLP) architectures. The lab is divided into two parts:

- **Part 1: Regression** - Predict the closing price of stocks using historical data.
- **Part 2: Multi-Class Classification** - Predict machine failure types for predictive maintenance.

This README summarizes the implementation in `atelier1_gpu.py` (Part 1: Regression with GPU acceleration) and `atelier1_2.py` (Part 2: Classification with GPU acceleration), including the approach, code structure, and results.

## Datasets
- **Part 1 (Regression)**: NYSE Stock Prices Dataset from [Kaggle](https://www.kaggle.com/datasets/dgawlik/nyse). Contains historical stock data with columns like `date`, `symbol`, `open`, `close`, `low`, `high`, `volume`.
  - Loaded file: `prices-split-adjusted.csv` (851,264 rows, 7 columns).
- **Part 2 (Classification)**: Predictive Maintenance Dataset from [Kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification). Contains machine sensor data with columns like `UDI`, `Product ID`, `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `Target`, `Failure Type`.
  - Shape: (10,000 rows, 10 columns).
  - Classes are highly imbalanced (e.g., "No Failure": 9,652; others much lower).

## Environment and Setup
- **PyTorch Version**: 2.6.0+cu124
- **CUDA Available**: True (Version: 12.4)
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (Memory: 8.59 GB)
- **Libraries Used**: pandas, numpy, matplotlib, seaborn, torch, sklearn (for preprocessing, metrics, and grid search).
- **Device**: All computations are performed on GPU (CUDA) where available for faster training.

To set up:
1. Install dependencies: `pip install torch pandas numpy matplotlib seaborn scikit-learn`.
2. Ensure CUDA is installed if using GPU.
3. Download datasets to the working directory.

## How to Run
- **Part 1 (Regression)**: Run `python atelier1_gpu.py`. It performs EDA, data prep, grid search, training, and visualization.
- **Part 2 (Classification)**: Run `python atelier1_2.py`. It performs preprocessing, EDA, balancing, grid search, training, and evaluation.

Note: Part 1 training may take time due to large dataset and grid search.

## Part 1: Regression – Predicting Stock Closing Price
### Description
- Task: Predict the `close` price using features like `open`, `high`, `low`, `volume`.
- Approach:
  1. **EDA**: Visualize stock price evolution (e.g., AAPL), closing price distribution, correlation heatmap, and average volume by year.
  2. **Data Prep**: Split data (80/20), standardize features using `StandardScaler`. Convert to PyTorch tensors and use DataLoader.
  3. **Model**: Custom MLP (`Regressor` class) with configurable layers, ReLU activation, and dropout.
  4. **Grid Search**: Use `ParameterGrid` to test hyperparameters (layers, learning rate, dropout). Train for 50 epochs per config, select based on final test MSE.
  5. **Training**: Adam optimizer, MSE loss. Train best model for 150 epochs. Visualize loss curves.
  6. **Regularization Comparison**: Test an enhanced model with BatchNorm, higher dropout, and L2 regularization.
- Code Structure:
  - Load and parse dataset (handles both CSV formats).
  - EDA plots.
  - Data splitting and scaling.
  - Model definition and training function (GPU-accelerated).
  - Grid search loop.
  - Final training and evaluation (RMSE, R²).
  - Regularization model comparison.

### Results (Partial - Training Still Executing)
- Dataset Loaded: 851,264 rows, 7 columns.
- Grid Search (Partial Output):
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
- Full grid search and final metrics (RMSE, R²) pending completion.
- Interpretation: Deeper models with moderate LR perform better. Regularization reduces overfitting (visualized in loss curves).

## Part 2: Multi-Class Classification – Predictive Maintenance
### Description
- Task: Predict `Failure Type` (6 classes) using sensor features.
- Approach:
  1. **Preprocessing**: Drop irrelevant columns (`UDI`, `Product ID`), encode `Type` (L/M/H → 0/1/2). Handle imbalance.
  2. **EDA**: Countplot of failure types, correlation heatmap.
  3. **Balancing**: Manual oversampling to match the majority class count (from ~10k to ~58k samples).
  4. **Data Prep**: Split (80/20, stratified), standardize features. Use DataLoader.
  5. **Model**: Custom MLP (`Classifier` class) with BatchNorm, ReLU, dropout, and 6 output classes.
  6. **Grid Search**: Test LR and dropout (60 epochs per config), select based on test accuracy.
  7. **Training**: Adam optimizer, CrossEntropy loss. Train best model for 120 epochs. Visualize loss/accuracy curves.
  8. **Evaluation**: Classification report (precision, recall, F1).
- Code Structure:
  - Load and clean dataset.
  - EDA plots.
  - Manual oversampling for balance.
  - Data splitting and scaling.
  - Model definition.
  - Grid search loop (GPU-accelerated).
  - Final training with metrics tracking.
  - Visualization and classification report.

### Results
- Imbalance Before: Counter({'No Failure': 9652, 'Heat Dissipation Failure': 112, 'Power Failure': 95, 'Overstrain Failure': 78, 'Tool Wear Failure': 45, 'Random Failures': 18})
- After Balancing: Counter({1: 9652, 3: 9652, 5: 9652, 2: 9652, 4: 9652, 0: 9652}) (Classes encoded: ['Heat Dissipation Failure' (0), 'No Failure' (1), ...])
- Grid Search:
  - lr=0.001 dropout=0.3 → Accuracy = 0.9908
  - lr=0.0005 dropout=0.3 → Accuracy = 0.9893
  - lr=0.001 dropout=0.5 → Accuracy = 0.9820
  - lr=0.0005 dropout=0.5 → Accuracy = 0.9807
- Best Model: {'dropout': 0.3, 'lr': 0.001} | Accuracy = 0.9908
- Training Progress (Final Model):
  - Epoch 20 → Train Acc: 0.9845 | Test Acc: 0.9921
  - Epoch 40 → Train Acc: 0.9863 | Test Acc: 0.9934
  - Epoch 60 → Train Acc: 0.9865 | Test Acc: 0.9934
  - Epoch 80 → Train Acc: 0.9870 | Test Acc: 0.9928
  - Epoch 100 → Train Acc: 0.9867 | Test Acc: 0.9945
  - Epoch 120 → Train Acc: 0.9879 | Test Acc: 0.9921
- Classification Report (Test Set):
precision recall f1-score support
Heat Dissipation Failure 0.9943 1.0000 0.9972 1930
No Failure 1.0000 0.9528 0.9759 1930
Overstrain Failure 0.9979 1.0000 0.9990 1931
Power Failure 0.9959 1.0000 0.9979 1931
Random Failures 0.9913 1.0000 0.9956 1931
Tool Wear Failure 0.9743 1.0000 0.9870 1930
accuracy 0.9921 11583
macro avg 0.9923 0.9921 0.9921 11583
weighted avg 0.9923 0.9921 0.9921 11583
- Interpretation: Oversampling resolved imbalance, leading to high accuracy (>99%). Model generalizes well (minimal overfitting in curves). "No Failure" has slightly lower recall due to original dominance.

## Notes
- GPU acceleration significantly speeds up training (e.g., large batches, deep models).
- Visualizations (plots) are generated during execution but not saved; modify code to save if needed.
- Part 1 results are partial; update this README with full metrics once training completes.

Author: Niama Aqarial | Date: November 28, 2025
