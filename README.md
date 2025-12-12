House Price Predictor
A machine learning project implementing linear regression from scratch using NumPy and comparing it with scikit-learn's implementation on the Kaggle House Prices dataset.
Project Overview
This project demonstrates a complete understanding of linear regression by:

Building gradient descent algorithm from scratch
Training on real-world housing data
Comparing custom implementation with industry-standard sklearn
Visualizing training progress and model performance

Objectives

Implement linear regression without ML libraries (using only NumPy)
Understand the mathematics behind gradient descent
Track and visualize model training (loss curves)
Validate custom implementation against sklearn
Practice proper ML workflow (train/test split, normalization, evaluation)

Dataset
Source: Kaggle House Prices Dataset
Feature Used: OverallQual (Overall material and finish quality)
Target: SalePrice (House sale price in dollars)
Samples: 1460 houses (after removing missing values)
Split: 80% training, 20% testing
🔧 Technical Implementation
Manual Implementation

Algorithm: Gradient Descent
Loss Function: Mean Squared Error (MSE)
Parameters: Slope (m) and Intercept (b)
Hyperparameters:

Learning Rate: 0.05
Iterations: 5000


Preprocessing: Min-max normalization on features

Key Components
Gradient Calculation:
python# Gradient for slope (m)
dJ_dm = mean(2 * X * (m*X + b - y))

# Gradient for intercept (b)
dJ_db = mean(2 * (m*X + b - y))
Parameter Update:
pythonm = m - learning_rate * dJ_dm
b = b - learning_rate * dJ_db
Results
Model Performance
ImplementationTrain MSETest MSEManual (Gradient Descent)~2.09e+09~2.38e+09Scikit-learn~2.09e+09~2.38e+09
Key Findings:

Manual implementation converges successfully
Results match sklearn within acceptable margin
Loss decreases consistently over 5000 iterations
Model generalizes reasonably to test set

Visualizations
1. Loss Curve

Shows MSE decreasing over 5000 iterations
Both train and test loss tracked
Demonstrates successful convergence

2. Predictions Comparison

Scatter plots comparing actual vs predicted prices
Visual comparison of manual model vs sklearn
Shows model learns the underlying relationship

Key Insights
What I Learned

MSE Penalizes Large Errors
Squaring the errors in MSE means that larger prediction mistakes are penalized exponentially more than small ones. This forces the model to prioritize fixing big errors first.
Gradient Descent is Iterative Optimization
The algorithm doesn't solve for the best parameters directly—it takes small steps in the direction that reduces error, gradually finding the minimum loss.
Learning Rate Matters
Too high (>0.1): Model might overshoot and diverge
Too low (<0.01): Training takes unnecessarily long
Sweet spot (0.05): Fast convergence while remaining stable
Normalization Improves Training
Scaling features to [0,1] range makes gradient descent converge faster and more reliably, especially important when features have different scales.
Train/Test Split Validates Generalization
Tracking both train and test MSE helps detect overfitting. In this project, both curves decrease together, indicating good generalization.

How to Run
Prerequisites
bashpip install numpy pandas matplotlib scikit-learn
File Structure
house-price-predictor/
├── data/
│   └── train.csv
├── notebooks/
│   └── linear_regression_analysis.ipynb
├── results/
│   ├── loss_curve.png
│   └── predictions_comparison.png
└── README.md
Running the Project

Clone the repository
Place the Kaggle dataset in data/ folder
Open linear_regression_analysis.ipynb
Run all cells sequentially
Check results/ folder for saved visualizations

Code Highlights
Custom Gradient Descent Loop
pythonfor iteration in range(5000):
    # Calculate predictions
    y_pred = m * X_train + b
    
    # Compute gradients
    dJ_dm = np.mean(2 * X_train * (y_pred - y_train))
    dJ_db = np.mean(2 * (y_pred - y_train))
    
    # Update parameters
    m = m - learning_rate * dJ_dm
    b = b - learning_rate * dJ_db
    
    # Track loss
    mse = np.mean((y_pred - y_train)**2)
    loss_history.append(mse)
Sklearn Comparison
pythonfrom sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred_sklearn = model.predict(X_test)
mse_sklearn = np.mean((y_pred_sklearn - y_test)**2)
Technologies Used

Python 3.x
NumPy - Numerical computations and gradient descent
Pandas - Data loading and preprocessing
Matplotlib - Visualization and plotting
Scikit-learn - Model validation and comparison

Concepts Demonstrated

Linear Regression fundamentals
Gradient Descent optimization
Mean Squared Error (MSE) loss function
Feature normalization (min-max scaling)
Train/test data splitting
Model evaluation and comparison
Data visualization for ML

Future Improvements

 Add multiple features (multivariate regression)
 Implement regularization (Ridge/Lasso)
 Add cross-validation for robust evaluation
 Experiment with different learning rates
 Add early stopping to prevent overfitting
 Try polynomial features for non-linear relationships

Acknowledgments

Dataset: Kaggle House Prices Competition
Inspiration: Understanding ML fundamentals from first principles
Goal: Build strong foundation before using high-level libraries

Contact
