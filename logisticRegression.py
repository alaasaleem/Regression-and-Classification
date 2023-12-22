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

# Learn a logistic regression model with a linear decision boundary
model = LogisticRegression()
model.fit(X_train, class_train)

# Make predictions on the test data and evaluate accuracy
class_pred = model.predict(X_test)
accuracy = accuracy_score(class_test, class_pred)
print(f"Accuracy: {accuracy}")

# Set up plot with specific style
plt.figure(figsize=(8, 6), facecolor='black')

# Scatterplot of training data points with labels for C1 (green) and C2 (orange)
c1_indices = class_train == 'C1'
c2_indices = class_train == 'C2'

plt.scatter(X_train[c1_indices, 0], X_train[c1_indices, 1], c='green', label='Class C1')
plt.scatter(X_train[c2_indices, 0], X_train[c2_indices, 1], c='orange', label='Class C2')

# Decision boundary line equation
w0 = model.intercept_[0]
w1, w2 = model.coef_.T
x1_min, x1_max = X_train[:, 0].min(), X_train[:, 0].max()
x1 = np.linspace(x1_min, x1_max, 100)
x2 = -(w0 + w1 * x1) / w2  # Line equation (w0 + w1*x1 + w2*x2 = 0)

# Plotting decision boundary with custom styling and add label
plt.plot(x1, x2, 'red', linestyle='-', lw=1, label='Linear Decision Boundary')  # Label for the red line

# Get the coefficients and intercept
coefficients = model.coef_.flatten()
intercept = model.intercept_[0]

# Display the learned equation
equation = f"y = {-(intercept/coefficients[1]):.2f} - ({coefficients[0]/coefficients[1]:.2f})x1"
plt.annotate(equation, xy=(0.5, 0.5), xycoords='axes fraction', color='white', fontname='Times New Roman',
             xytext=(0.387, 0.5), 
             arrowprops=dict(color='white', arrowstyle='<-', connectionstyle='arc3,rad=-0.7'))

# Plot settings and legend
plt.xlabel('x1', fontdict={'fontname': 'Times New Roman', 'fontsize': 14, 'color': 'white'})
plt.ylabel('x2', fontdict={'fontname': 'Times New Roman', 'fontsize': 14, 'color': 'white'})
plt.title('Logistic Regression with a Linear Decision Boundary', fontdict={'fontname': 'Times New Roman', 'fontsize': 18, 'color': 'white'})
plt.xticks(color='white', fontname='Times New Roman')
plt.yticks(color='white', fontname='Times New Roman')
plt.grid(True, color='white')
plt.gca().set_facecolor('black')

# Adding legend for scatter points and red line
legend = plt.legend(loc='upper right', fontsize=12, facecolor='black', edgecolor='white', prop={'family': 'Times New Roman', 'weight': 'bold'})
for text in legend.get_texts():
    text.set_color('white')

# Show plot
plt.show()