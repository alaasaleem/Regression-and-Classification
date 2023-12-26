# Model Selection and Hyperparameters Tuning

This project involves implementing regression and classification techniques using Python. The dataset provided (`data_reg.csv`) comprises 200 examples, each containing attributes `x1` and `x2`, along with a continuous target label `y`.

## Task 1: Data Preparation and Visualization

- **Data Splitting:** Read `data_reg.csv` and split it into training, validation, and testing sets (120, 40, and 40 examples, respectively).
- **Visualization:** Generate a 3D scatter plot, plotting `x1` and `x2` on the x and y axes, respectively. Encode the target label `y` as the z-axis, using different colors for each set.

## Task 2: Polynomial Regression

- **Modeling:** Apply polynomial regression (degrees 1 to 10) on the training set.
- **Validation:** Determine the best polynomial degree by plotting the validation error versus polynomial degree curve.
- **Visualization:** Plot the surface of the learned function alongside training examples for each polynomial degree.

## Task 3: Ridge Regression

- **Implementation:** Implement ridge regression with a polynomial of degree 8 on the training set.
- **Parameter Tuning:** Choose the best regularization parameter from options: {0.001, 0.005, 0.01, 0.1, 10}.
- **Evaluation:** Plot the Mean Squared Error (MSE) on the validation set versus the regularization parameter.

## Logistic Regression

For the classification task, the dataset includes `train_cls.csv` for training and `test_cls.csv` for testing.

### Task 1: Linear Decision Boundary

- **Model Learning:** Utilize scikit-learn's logistic regression to learn a model with a linear decision boundary.
- **Visualization:** Plot the decision boundary on a scatter plot of the training set.
- **Accuracy:** Calculate training and testing accuracies for the learned model.

### Task 2: Quadratic Decision Boundary

- **Modeling:** Repeat Task 1 but with a quadratic decision boundary.

### Task 3: Model Evaluation

Comment on the learned models from Tasks 1 and 2, discussing potential overfitting or underfitting concerns.

---
