import pandas as pd

data = pd.read_csv('Housing.csv')
X = data.drop("price", axis=1)
y = data["price"]

print(X.columns)