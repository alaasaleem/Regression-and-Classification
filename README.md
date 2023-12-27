# Regression and Classification
This project involves implementing regression and classification techniques using Python.

> This project is an assignment of the **Machine Learning and Data Science (ENCS5341) course** at [Birzeit University](https://www.birzeit.edu).

## Part 1: Model Selection and Hyperparameters Tuning

The dataset provided (`data_reg.csv`) contains 200 examples, each containing attributes `x1` and `x2`, corresponding with a continuous target label `y`.

### Task 1: Data Splitting and Visualization

- **Data Splitting:** Read `data_reg.csv` and split it into training, validation, and testing sets.
- **Visualization:** Generate a 3D scatter plot showing the distribution of the three sets.
> *3D scatter plot:*
  ![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/8100e3ca-1099-4b5d-bc26-9c70d2f377d3)

### Task 2: Polynomial Regression

- **Modeling:** Apply polynomial regression (degrees 1 to 10) on the training set.
- **Validation:** Justify the best polynomial degree via a validation error vs polynomial degree curve.

> *The Best Polynomial Degree:*

  ![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/1897923d-b2d2-4e79-8b4c-baa9f7ca87b2)

> *Validation Error vs Polynomial Degree Curve:*
![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/df39d381-6be6-4860-a55d-fc6aea91253a)

> *Degree Surface Results:*
> *Degree 1*
  ![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/c38fa326-47f3-4a08-9f9e-a1e695893634)

> *Degree 2*
  ![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/7ff6df08-cf92-4976-8e8b-c94500d12276)

> *Degree 3*
  ![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/e19e8f6f-baba-4a28-9e94-d8256716e78c)

> *Degree 4*
  ![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/8acf0612-8406-4531-b7b3-a3856bc0e35d)

> *Degree 5*
  ![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/bc3d345e-2afc-4796-82cb-870cd01c4718)

> *Degree 6*
  ![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/03752f53-a8da-4229-bfd5-6fdceb9276b4)

> *Degree 7*
  ![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/5d072a1f-b765-4cb0-9824-7e02be9d2fb7)

> *Degree 8*
  ![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/9714d545-1445-4c70-bb3c-9d0fb6a3af21)

> *Degree 9*
  ![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/35f0591e-26aa-4674-880f-bacc7e67d2c1)

> *Degree 10*
  ![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/1d2b7b3f-844d-44cc-8f9c-e277edd21666)

### Task 3: Ridge Regression

- **Modeling:** Apply ridge regression with a polynomial of degree 8 on the training set.
- **Parameter Tuning:** Choose the best regularization parameter.
> *The Best Regularization Parameter:*

  ![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/2b1ba69b-7030-4869-b663-efb045054cc1)

- **Visualization:** Plot the Mean Squared Error (MSE) vs the regularization parameter.
> *MSE vs Regularization Parameter Plot:*
  ![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/d1fa71cd-d432-4d7b-9b0f-2f8fc340e8c5)

## Part 2: Logistic Regression

For the classification task, the dataset includes `train_cls.csv` for training and `test_cls.csv` for testing.

### Task 1: Linear Decision Boundary

- **Modeling:** Apply logistic regression to learn a model with a linear decision boundary.
- **Visualization:** Show the decision boundary on a scatter plot of the training set.
> *Linear Decision Boundary Plot with Accuracy results:*
  ![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/e50a7f6b-df5a-463d-ae98-22a0e98b3721)

### Task 2: Quadratic Decision Boundary

- **Modeling:** Repeat Task 1 but with a quadratic decision boundary.
> *Quadratic Decision Boundary Plot with Accuracy results:*
  ![image](https://github.com/alaasaleem/Hyperparameter-Tuning/assets/127680801/b2728de9-fbac-408c-9803-bf22eacee0ae)

## Usage

### Obtaining the Files

**Clone the Repository:**
   ```bash
   git clone https://github.com/alaasaleem/Hyperparameter-Tuning
   cd Hyperparameter-Tuning
   ```
### Running Scripts

*Ensure you have Python installed*

*Ensure data_reg.csv, train_cls.csv, and test_cls.csv are in the same directory as the scripts.*

**Run the Polynomial Regression Script:**
  ```bash
  python polynomialRegrission.py
  ```

**Run the Ridge Regression Script:**
  ```bash
  python ridgeRegression.py
  ```

**Run the Linear Logistic Regression Script:**
  ```bash
  python logisticRegressionLinear.py
  ```

**Run the Non-Linear Logistic Regression Script:**
  ```bash
  python logisticRegressionNonLinear.py
  ```
