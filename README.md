# Regression and Classification
This project involves implementing regression and classification techniques using Python.

> This project is an assignment of the **Machine Learning and Data Science (ENCS5341) course** at [Birzeit University](https://www.birzeit.edu).

## Part 1: Model Selection and Hyperparameters Tuning

The dataset provided (`data_reg.csv`) contains 200 examples, each containing attributes `x1` and `x2`, corresponding with a continuous target label `y`.

### Task 1: Data Splitting and Visualization

- **Data Splitting:** Read `data_reg.csv` and split it into training, validation, and testing sets.
- **Visualization:** Generate a 3D scatter plot showing the distribution of the three sets.
- **3D scatter plot:**
![video](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/d884345d-4944-4bc4-963d-c6c249f80dd4)


### Task 2: Polynomial Regression

- **Modeling:** Apply polynomial regression (degrees 1 to 10) on the training set.
- **Validation:** Justify the best polynomial degree via a validation error vs polynomial degree curve.
- **Validation Error vs Polynomial Degree Curve:**
![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/3aecbdb2-1cc9-4ac9-ab60-4ae59329a899)

- **Degree Surface Results:** 
> Degree 1
![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/c38fa326-47f3-4a08-9f9e-a1e695893634)


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
