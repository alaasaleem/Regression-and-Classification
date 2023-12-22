import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Read the data
test_dataFrame = pd.read_csv("C:\\Users\\User\\OneDrive\\BZU\\120-07\\ML\\Assignment2\\Hyperparameter-Tuning\\test_cls.csv")
X_test = test_dataFrame[['x1', 'x2']].values
class_test = test_dataFrame['class'].values

train_dataFrame = pd.read_csv("C:\\Users\\User\\OneDrive\\BZU\\120-07\\ML\\Assignment2\\Hyperparameter-Tuning\\train_cls.csv")
X_train = train_dataFrame[['x1', 'x2']].values
class_train = train_dataFrame['class'].values

# Create polynomial features and transform the input data
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Learn a logistic regression model with a quadratic decision boundary
model = LogisticRegression()
model.fit(X_train_poly, class_train)

# Compute training accuracy
train_accuracy = model.score(X_train_poly, class_train)

# Make predictions on the test data and evaluate accuracy
class_pred = model.predict(X_test_poly)
test_accuracy = accuracy_score(class_test, class_pred)

# Set up plot with specific style
plt.figure(figsize=(8, 6), facecolor='black')

# Scatterplot of training data points with labels for C1 (green) and C2 (orange) with different markers
c1_indices = class_train == 'C1'
c2_indices = class_train == 'C2'
plt.scatter(X_train[c1_indices, 0], X_train[c1_indices, 1], c='red', label='Class C1', marker='s')
plt.scatter(X_train[c2_indices, 0], X_train[c2_indices, 1], c='green', label='Class C2', marker='^')

# Decision boundary curve equation
w = model.coef_[0]
b = model.intercept_[0]
x1_min, x1_max = X_train[:, 0].min(), X_train[:, 0].max()
x2_min, x2_max = X_train[:, 1].min(), X_train[:, 1].max()
x1 = np.linspace(x1_min, x1_max, 100)
x2 = np.linspace(x2_min, x2_max, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = b + w[1] * X1 + w[2] * X2 + w[3] * X1**2 + w[4] * X1 * X2 + w[5] * X2**2

# Plotting decision boundary with custom styling and add label
plt.contour(X1, X2, Z, levels=[0], colors='blue', linestyles='--', linewidths=1, label='Quadratic Decision Boundary')

equation = f"y = {b:.2f} + ({w[1]:.2f})x1 + ({w[2]:.2f})x2 + ({w[3]:.2f})x1^2 + ({w[4]:.2f})x1x2 + ({w[5]:.2f})x2^2"
plt.text(x1_max - 2.3, x2_min + 0.8, equation, fontsize=10, color='green', fontname='Times New Roman',
         bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.5'))

# Display accuracy on the plot
train_accuracy_text = f"Training Accuracy: {train_accuracy * 100:.2f}%"
test_accuracy_text = f"Testing Accuracy: {test_accuracy * 100:.2f}%"

plt.text(x1_min - 0.5, x2_max - 0.6, train_accuracy_text, ha='left', fontsize=10, color='green', fontname='Times New Roman', 
         bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.5'))  

plt.text(x1_min - 0.49, x2_max - 0.76, test_accuracy_text, ha='left', fontsize=10, color='green', fontname='Times New Roman', 
         bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.5'))  


# Plot settings, legend, axis limits, and display plot
plt.xlabel('x1', fontdict={'fontname': 'Times New Roman', 'fontsize': 14, 'color': 'white'})
plt.ylabel('x2', fontdict={'fontname': 'Times New Roman', 'fontsize': 14, 'color': 'white'})
plt.title('Logistic Regression with a Quadratic Decision Boundary', fontdict={'fontname': 'Times New Roman', 'fontsize': 18, 'color': 'white'})
plt.xticks(color='white', fontname='Times New Roman')
plt.yticks(color='white', fontname='Times New Roman')
plt.grid(True, color='white')
plt.gca().set_facecolor('black')

legend = plt.legend(loc='upper right', fontsize=12, facecolor='black', edgecolor='white', prop={'family': 'Times New Roman', 'weight': 'bold'})
for text in legend.get_texts():
    text.set_color('white')

plt.xlim(x1_min - 0.5, x1_max + 0.5)
plt.ylim(x2_min - 0.5, x2_max + 0.5)

plt.show()
