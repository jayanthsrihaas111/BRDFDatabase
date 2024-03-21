import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('/Users/jayanthsrihaas111/Desktop/BRDFDatabase/output.csv')

# Assuming the last column is the target and the rest are features
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Normalize or scale X if necessary

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

