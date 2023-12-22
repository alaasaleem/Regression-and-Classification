import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Read the data
test_dataFrame = pd.read_csv("C:\\Users\\User\\OneDrive\\BZU\\120-07\\ML\\Assignment2\\Hyperparameter-Tuning\\test_cls.csv")
X_test = test_dataFrame[['x1', 'x2']].values
class_test = test_dataFrame['class'].values

train_dataFrame = pd.read_csv("C:\\Users\\User\\OneDrive\\BZU\\120-07\\ML\\Assignment2\\Hyperparameter-Tuning\\train_cls.csv")
X_train = train_dataFrame[['x1', 'x2']].values
class_train = train_dataFrame['class'].values

# Create logistic regression model 
model = LogisticRegression()
model.fit(X_train, class_train)

# Find training accuracy
train_accuracy = model.score(X_train, class_train)

# Predict the test data and find accuracy
class_pred = model.predict(X_test)
test_accuracy = accuracy_score(class_test, class_pred)

plt.figure(figsize=(8, 6), facecolor='black')

c1_indices = class_train == 'C1'
c2_indices = class_train == 'C2'

plt.scatter(X_train[c1_indices, 0], X_train[c1_indices, 1], c='red', label='Class C1', marker='s')  
plt.scatter(X_train[c2_indices, 0], X_train[c2_indices, 1], c='green', label='Class C2', marker='^')  

# Line equation (w0 + w1*x1 + w2*x2 = 0)
w0 = model.intercept_[0]
w1, w2 = model.coef_.T
x1_min, x1_max = X_train[:, 0].min(), X_train[:, 0].max()
x1 = np.linspace(x1_min, x1_max, 100)
x2 = -(w0 + w1 * x1) / w2  

# Plot decision boundary
plt.plot(x1, x2, 'blue', linestyle='-', lw=1, label='Linear Decision Boundary')  # Label for the red line

# The coefficients and intercept
coefficients = model.coef_.flatten()
intercept = model.intercept_[0]

# Show the learned equation
equation = f"y = {-(intercept/coefficients[1]):.2f} - ({coefficients[0]/coefficients[1]:.2f})x1"
plt.text(-0.97, 1.6, equation, color='white', fontname='Times New Roman', bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.5'))


# Show accuracy on the plot
train_accuracy_text = f"Training Accuracy: {train_accuracy * 100:.2f}%"
test_accuracy_text = f"Testing Accuracy: {test_accuracy * 100:.2f}%"

plt.text(-0.74, 1.13, train_accuracy_text, ha='right', fontsize=10, color='green', fontname='Times New Roman', 
         bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.5'))

plt.text(-0.75, 1.34, test_accuracy_text, ha='right', fontsize=10, color='green', fontname='Times New Roman', 
         bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.5'))


plt.xlabel('x1', fontdict={'fontname': 'Times New Roman', 'fontsize': 14, 'color': 'white'})
plt.ylabel('x2', fontdict={'fontname': 'Times New Roman', 'fontsize': 14, 'color': 'white'})
plt.title('Logistic Regression with a Linear Decision Boundary', fontdict={'fontname': 'Times New Roman', 'fontsize': 18, 'color': 'white'})
plt.xticks(color='white', fontname='Times New Roman')
plt.yticks(color='white', fontname='Times New Roman')
plt.grid(True, color='white')
plt.gca().set_facecolor('black')

legend = plt.legend(loc='upper right', fontsize=12, facecolor='black', edgecolor='white', prop={'family': 'Times New Roman', 'weight': 'bold'})
for text in legend.get_texts():
    text.set_color('white')

plt.show()
