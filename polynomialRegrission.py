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

for degree in range(1, 11):
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly_train = poly_reg.fit_transform(X_train)
    X_poly_val = poly_reg.transform(X_val)
    
    poly_model = LinearRegression()
    poly_model.fit(X_poly_train, y_train)

    y_train_pred = poly_model.predict(X_poly_train)
    y_val_pred = poly_model.predict(X_poly_val)
    
    train_error = mean_squared_error(y_train, y_train_pred)
    val_error = mean_squared_error(y_val, y_val_pred)

    train_errors.append(train_error)
    val_errors.append(val_error)

    if val_error < best_val_error:
        best_val_error = val_error
        best_degree = degree

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    examples = [
        (X_train, y_train, 'red', 'Training Examples'),
        (X_val, y_val, 'green', 'Validation Examples'),
    ]

    for X, y, color, label in examples:
        ax.scatter(X[:, 0], X[:, 1], y, c=color, label=label, marker='o', s=50, alpha=0.8)

    # Generating the polynomial curve
    x1, x2 = np.meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100),
                         np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100))
    X_mesh = np.c_[x1.ravel(), x2.ravel()]
    X_mesh_poly = poly_reg.transform(X_mesh)
    y_mesh = poly_model.predict(X_mesh_poly).reshape(x1.shape)
    
    ax.plot_surface(x1, x2, y_mesh, alpha=0.5, color='blue')  # Plotting the polynomial curve

    ax.set_xlabel('x1', fontsize=12, fontname='Times New Roman', color='white')
    ax.set_ylabel('x2', fontsize=12, fontname='Times New Roman', color='white')
    ax.set_zlabel('y', fontsize=12, fontname='Times New Roman', color='white')
    ax.set_title(f'Degree {degree} Polynomial', fontdict={'fontname': 'Times New Roman', 'fontsize': 27, 'color': 'white'})

    legend = ax.legend(bbox_to_anchor=(0.05, 0.6), fontsize=12, facecolor='black', edgecolor='white',
                   prop={'family': 'Times New Roman', 'weight': 'bold'})

    for text in legend.get_texts(): 
        text.set_color('white')

    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo['grid'].update(color='white')
        [tick.set(fontname="Times New Roman", fontsize=12, color='white') for tick in axis.get_ticklabels()]

    comment = ""
    if (train_error < 0.5 and val_error > 0.5):
        comment = "**Overfitting**: Low train error, high validation error. Model captures noise in data."
    elif (train_error > 0.5 and val_error > 0.5):
        comment = "**Underfitting**: High train & validation error. Model is too simple."
    else:
        comment = "**Good Fitting**: Low train & validation error. Model generalizes well."

    ax.text2D(0.05, 0.85, comment, color='green' if train_error < 0.5 and val_error < 0.5 else 'red',
          transform=ax.transAxes, fontweight='bold', fontsize=12, fontname='Times New Roman')

    ax.text2D(0.35, 0.88, f'Training Error: {train_error:.4f}\nValidation Error: {val_error:.4f}',
              transform=ax.transAxes, fontweight='bold', fontsize=12, fontname='Times New Roman', color='white')
    
    plt.show()

print(f"The best fitting model is Degree {best_degree} Polynomial with a validation error of {best_val_error:.4f}")

# Plotting validation error vs polynomial degree curve
degrees = range(1, 11)
# Find the index of the minimum validation error
min_val_error_idx = np.argmin(val_errors)
min_val_error_degree = degrees[min_val_error_idx]
min_val_error = val_errors[min_val_error_idx]

plt.figure(figsize=(8, 6), facecolor='black')
plt.plot(degrees, val_errors, marker='o', linestyle='-', color='red')
plt.scatter(min_val_error_degree, min_val_error, color='green', s=100, label=f'Lowest Error ({min_val_error:.4f})')

plt.annotate(f'Minimum Error: Degree {min_val_error_degree}', 
             xy=(min_val_error_degree, min_val_error), 
             xytext=(min_val_error_degree - 1.5, min_val_error + 0.1), 
             arrowprops=dict(color='white', arrowstyle='->', connectionstyle='arc3,rad=.5'),
             fontsize=12, fontname='Times New Roman', color='white')

plt.title('Validation Error vs Polynomial Degree', fontdict={'fontname': 'Times New Roman', 'fontsize': 18, 'color': 'white'})
plt.xlabel('Polynomial Degree', fontdict={'fontname': 'Times New Roman', 'fontsize': 14, 'color': 'white'})
plt.ylabel('Validation Error', fontdict={'fontname': 'Times New Roman', 'fontsize': 14, 'color': 'white'})
plt.xticks(degrees, color='white', fontname='Times New Roman')
plt.yticks(color='white', fontname='Times New Roman')
plt.grid(True, color='white')
plt.gca().set_facecolor('black')

legend = plt.legend(loc='upper right', fontsize=12, facecolor='black', edgecolor='white', prop={'family': 'Times New Roman', 'weight': 'bold'})
for text in legend.get_texts(): 
    text.set_color('white')

plt.show()