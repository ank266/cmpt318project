import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV


file_path = 'NGS-81M0011-E-2018_F1.csv'
data = pd.read_csv(file_path)

# print(data.head())

# print(data.columns.tolist())

data = data.dropna()

data = pd.get_dummies(data, drop_first=True)

features = ['PGM_034', 'PGM_P405', 'PGM_350', 'DDIS_FL']
target = 'AFT_P040'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

param_grid = {
    'C': [0.1, 1, 10, 100],
    'max_iter': [100, 200, 300]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
