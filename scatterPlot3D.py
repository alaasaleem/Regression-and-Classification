import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read the data from the csv file
dataFrame = pd.read_csv("C:\\Users\\User\\OneDrive\\BZU\\120-07\\ML\\Assignment2\\Hyperparameter-Tuning\\data_reg.csv")
X = dataFrame[['x1', 'x2']].values  # Features x1 and x2
y = dataFrame['y'].values  # Target variable y

# Split data into training set (the first 120 examples), validation set (the next 40 examples), and testing set (the last 40 examples)
X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=40, shuffle=False) #split the data set into main (160 examples) and test (last 40 examples)
X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=40, shuffle=False) # split the main (160 examples) into training (first 120 examples) and validation (last 40 examples)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

examples = [
    (X_train, y_train, '#1f77b4', f'Training Set ({len(X_train)} examples)'),  # Training set examples
    (X_val, y_val, '#ff7f0e', f'Validation Set ({len(X_val)} examples)'),  # Validation set examples
    (X_test, y_test, '#2ca02c', f'Test Set ({len(X_test)} examples)')  # Test set examples
]

# Plot each set as a scatter plot with their colors, labels
for X, y, color, label in examples:
    ax.scatter(X[:, 0], X[:, 1], y, c=color, label=label, marker='o', s=50, alpha=0.8)
    
# Define the font style
font = {'fontname': 'Times New Roman', 'fontsize': 12, 'color': 'white'} 

# Set labels for x, y, and z axes
ax.set_xlabel('x1',**font) 
ax.set_ylabel('x2',**font) 
ax.set_zlabel('y',**font)  

# Add legend with the updated labels including the number of examples
legend = ax.legend(bbox_to_anchor=(0.05, 0.6), fontsize=12, facecolor='black', edgecolor='white',
                   prop={'family': 'Times New Roman', 'weight': 'bold'})

for text in legend.get_texts(): 
    text.set_color('white')

ax.set_facecolor('black') 
fig.patch.set_facecolor('black')  

ax.set_title('Scatter Plot of Training, Validation, and Test Sets', y=1.02,
             fontdict={'fontname': 'Times New Roman', 'fontsize': 27, 'color': 'white'}) 

# Setting the surface color to black
ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))  
ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0)) 
ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis._axinfo['grid'].update(color='white') 
    [tick.set(fontname="Times New Roman", fontsize=12, color='white') for tick in axis.get_ticklabels()] 

plt.show()