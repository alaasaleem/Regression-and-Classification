# Regression and Classification
This project involves implementing regression and classification techniques using Python.

> This project is an assignment of the **Machine Learning and Data Science (ENCS5341) course** at [Birzeit University](https://www.birzeit.edu).

## Part 1: Model Selection and Hyperparameters Tuning

The dataset provided (`data_reg.csv`) contains 200 examples, each containing attributes `x1` and `x2`, corresponding with a continuous target label `y`.

### Task 1: Data Splitting and Visualization

- **Data Splitting:** Read `data_reg.csv` and split it into training, validation, and testing sets.
- **Visualization:** Generate a 3D scatter plot showing the distribution of the three sets.
![Scatter Plot 3D](https://github-production-user-asset-6210df.s3.amazonaws.com/127680801/292963029-ebd02944-7e71-40c0-bfa5-16aa4c60bb36.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20231227%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231227T052659Z&X-Amz-Expires=300&X-Amz-Signature=ef6c254993460ff2a48d5ce60a5d25fda4c8e5691ca20b2b6e9d098b4da38dab&X-Amz-SignedHeaders=host&actor_id=127680801&key_id=0&repo_id=734243043)


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
