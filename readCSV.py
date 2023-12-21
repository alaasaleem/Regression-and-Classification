import pandas as pd
from sklearn.model_selection import train_test_split

#Read the data from the csv file
dataFrame = pd.read_csv("C:\\Users\\User\\OneDrive\\BZU\\120-07\\ML\\Assignment2\\Hyperparameter-Tuning\\data_reg.csv")
X = dataFrame[['x1', 'x2']].values  # Features x1 and x2
y = dataFrame['y'].values  # Target variable y

#Split data into training set (the first 120 examples), validation set (the next 40 examples), and testing set (the last 40 examples)
X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=40, shuffle=False) #split the data set into main (160 examples) and test (last 40 examples)
X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=40, shuffle=False) # split the main (160 examples) into training (first 120 examples) and validation (last 40 examples)



