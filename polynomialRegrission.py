import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the dataset
dataFrame = pd.read_csv("C:\\Users\\User\\OneDrive\\BZU\\120-07\\ML\\Assignment2\\Hyperparameter-Tuning\\data_reg.csv")
X = dataFrame[['x1', 'x2']].values  # Features x1 and x2
y = dataFrame['y'].values  # Target variable y

# Split data into training set (first 120 examples), validation set (next 40 examples), and testing set (last 40 examples)
X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=40, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=40, shuffle=False)

train_errors = []
val_errors = []

best_degree = None
best_val_error = float('inf')

# Loop on degrees of the polynomial from 1 to 10
for degree in range(1, 11):
    # Add a feature to know that it is a polynomial regression with a degree
    poly_reg = PolynomialFeatures(degree=degree)
    
    # Use the training set to transform it to new features by using the polynomial transformation on the original features (x, x^2, x^3, etc)
    X_poly_train = poly_reg.fit_transform(X_train)
    
    # Use the transform function to make the validation set look like the new features to be ready to predict
    X_poly_val = poly_reg.transform(X_val)

    # Create a Linear Regression object to apply the new features (which are polynomial and non-linear data that have a certain degree)
    poly_model = LinearRegression()
    
    # Train the model
    poly_model.fit(X_poly_train, y_train)

    # Predict the y values using the polynomial model
    y_train_pred = poly_model.predict(X_poly_train)
    y_val_pred = poly_model.predict(X_poly_val)
    
    # Find the mean squared error (MSE)
    train_error = mean_squared_error(y_train, y_train_pred)
    val_error = mean_squared_error(y_val, y_val_pred)

    train_errors.append(train_error)
    val_errors.append(val_error)

    # Check for the best fitting model among all degrees
    if val_error < best_val_error:
        best_val_error = val_error
        best_degree = degree

    # Plotting the surface for the current degree
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate meshgrid for the surface plot
    x1, x2 = np.meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100),
                         np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100))
    X_mesh = np.c_[x1.ravel(), x2.ravel()]
    X_mesh_poly = PolynomialFeatures(degree=degree).fit_transform(X_mesh)
    y_mesh = poly_model.predict(X_mesh_poly).reshape(x1.shape)

    # Plotting the surface
    ax.plot_surface(x1, x2, y_mesh, alpha=0.5)

    # Plot training and validation examples
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color='red', label='Training Examples')
    ax.scatter(X_val[:, 0], X_val[:, 1], y_val, color='green', label='Validation Examples')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title(f'Degree {degree} Polynomial')

    # Plotting training and validation errors on the plot
    ax.text2D(0.05, 0.95, f'Train Error: {train_error:.4f}\nValidation Error: {val_error:.4f}', transform=ax.transAxes)
    
    comment = ""
    if (train_error < 0.5 and val_error > 0.5 ):
        comment = "**Overfitting**: Low train error, high validation error. Model is too complex and captures noise in data."
    elif (train_error > 0.5 and val_error > 0.5 ):
        comment = "**Underfitting**: High train & validation error. Model is too simple and cannot capture the complexity of data."
    else:
        comment = "**Good Fitting**: Model generalizes well."

    ax.text2D(0.05, 0.85, comment, color='green' if train_error < 0.5 and val_error < 0.5 else 'red', transform=ax.transAxes, fontweight='bold')

    plt.legend()
    plt.show()

print(f"The best fitting model is Degree {best_degree} Polynomial with a validation error of {best_val_error:.4f}")