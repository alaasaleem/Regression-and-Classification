# Regression and Classification
This project involves implementing regression and classification techniques using Python.

> This project is an assignment of the **Machine Learning and Data Science (ENCS5341) course** at [Birzeit University](https://www.birzeit.edu).

## Part 1: Model Selection and Hyperparameters Tuning

The dataset provided (`data_reg.csv`) contains 200 examples, each containing attributes `x1` and `x2`, corresponding with a continuous target label `y`.

### Task 1: Data Splitting and Visualization

- **Data Splitting:** Read `data_reg.csv` and split it into training, validation, and testing sets.
- **Visualization:** Generate a 3D scatter plot showing the distribution of the three sets.
[Download 3D Scatter Plot Video](https://raw.githubusercontent.com/alaasaleem/Hyperparameter-Tuning/main/Scatter%20Plot%203D.mp4)


### Task 2: Polynomial Regression

- **Modeling:** Apply polynomial regression (degrees 1 to 10) on the training set.
- **Validation:** Justify the best polynomial degree via a validation error vs polynomial degree curve.
![Placeholder for Validation Error vs Polynomial Degree Curve](path/to/your/validation_error_curve.png)

- **Degree Surface Videos:** 
    - [Video for Degree 1](path/to/your/video_degree_1.mp4)
    - [Video for Degree 2](path/to/your/video_degree_2.mp4)
    - ... (up to Degree 10)

### Task 3: Ridge Regression

- **Modeling:** Apply ridge regression with a polynomial of degree 8 on the training set.
- **Parameter Tuning:** Choose the best regularization parameter.
- **Visualization:** Plot the Mean Squared Error (MSE) vs the regularization parameter.
![Placeholder for MSE vs Regularization Parameter Plot](path/to/your/MSE_vs_regularization.png)

## Part 2: Logistic Regression

For the classification task, the dataset includes `train_cls.csv` for training and `test_cls.csv` for testing.

### Task 1: Linear Decision Boundary

- **Modeling:** Apply logistic regression to learn a model with a linear decision boundary.
- **Visualization:** Show the decision boundary on a scatter plot of the training set.
![Placeholder for Linear Decision Boundary Plot](path/to/your/linear_decision_boundary.png)
