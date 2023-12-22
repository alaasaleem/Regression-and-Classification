import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Read the data
dataFrame = pd.read_csv("C:\\Users\\User\\OneDrive\\BZU\\120-07\\ML\\Assignment2\\Hyperparameter-Tuning\\data_reg.csv")
X = dataFrame[['x1', 'x2']].values
y = dataFrame['y'].values

# Split data
X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=40, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=40, shuffle=False)

# Transform features into polynomial features of degree 8
poly = PolynomialFeatures(degree=8)
X_train_poly, X_test_poly, X_poly_val = poly.fit_transform(X_train), poly.transform(X_test), poly.transform(X_val)

# List of alpha values to try
alphas = [0.001, 0.005, 0.01, 0.1, 10]

# RidgeCV with alphas and storing CV values
ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True).fit(X_train_poly, y_train)

# Access cross-validation errors for each alpha and print
for alpha, cv_error in zip(ridge_cv.alphas, ridge_cv.cv_values_.T):
    print(f"Alpha: {alpha}, Mean Cross-Validation Error: {cv_error.mean()}")

# Best alpha and its associated error
print("The best alpha value is:", ridge_cv.alpha_)
print("The mean squared error on the test data is:", mean_squared_error(y_val, ridge_cv.predict(X_poly_val)))

# Extract alpha values and corresponding MSE
alphas = ridge_cv.alphas
mse_values = [cv.mean() for cv in ridge_cv.cv_values_.T]

# Plotting MSE vs Alpha without scientific notation for alpha values
plt.figure(figsize=(8, 6), facecolor='black')
plt.plot(alphas, mse_values, marker='o', linestyle='-', color='red')
plt.xscale('log')  # Use a logarithmic scale for alpha values
plt.xlabel('Regularization Parameter (Alpha)', fontdict={'fontname': 'Times New Roman', 'fontsize': 14, 'color': 'white'})
plt.ylabel('Mean Squared Error (MSE)', fontdict={'fontname': 'Times New Roman', 'fontsize': 14, 'color': 'white'})
plt.title('Validation MSE vs Alpha', fontdict={'fontname': 'Times New Roman', 'fontsize': 18, 'color': 'white'})
plt.xticks(alphas, [str(a) for a in alphas], color='white', fontname='Times New Roman')
plt.yticks(color='white', fontname='Times New Roman')
plt.grid(True, color='white')
plt.gca().set_facecolor('black')

# Annotating the best alpha with an arrow
best_alpha = ridge_cv.alpha_
best_error = min(mse_values)
plt.scatter(best_alpha, best_error, color='green', s=100, label=f'Lowest Error ({best_error:.4f})')
plt.annotate(f'Best Alpha: {best_alpha}', 
             xy=(best_alpha, best_error), 
             xytext=(best_alpha / 0.9, best_error  + 0.05), 
             arrowprops=dict(color='white', arrowstyle='->', connectionstyle='arc3,rad=.9'),
             fontsize=12, fontname='Times New Roman', color='white')

legend = plt.legend(loc='upper right', fontsize=12, facecolor='black', edgecolor='white', prop={'family': 'Times New Roman', 'weight': 'bold'})
for text in legend.get_texts(): 
    text.set_color('white')
    
plt.show()

