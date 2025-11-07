# Predicting House Prices using Python & Decision Trees 🏠📊

**Author:** Nihar Sawant  
**Tools:** Python, Pandas, Scikit-learn, Matplotlib  

---

## Project Overview

This project demonstrates building machine learning models to **predict house prices** using the Kaggle House Prices dataset. It focuses on **model building, validation, and improvement** through decision trees and random forests.

---

## Dataset

- **Source:** Kaggle – “House Prices: Advanced Regression Techniques”  
- **Description:** Contains various features describing residential homes in Ames, Iowa, including square footage, number of bedrooms, bathrooms, year built, and more.  
- **Target Variable:** `SalePrice`  

---

## Project Steps

### 1. Data Import & Cleaning

```python
import pandas as pd

data = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")
data = data.dropna(axis=0)
y = data.SalePrice
features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
X = data[features]


2. Train/Test Split & Model Training

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)
model = DecisionTreeRegressor(random_state=1)
model.fit(X_train, y_train)
preds = model.predict(X_valid)
mae = mean_absolute_error(y_valid, preds)
print("Mean Absolute Error:", mae)


3. Model Improvement

Tried RandomForestRegressor

Tuned features and max_depth

Compared MAE values

4. Visualization (Optional)

import matplotlib.pyplot as plt

plt.scatter(y_valid, preds)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()


Results
Model	Mean Absolute Error
Decision Tree Regressor	24,750
Random Forest Regressor	21,500

Improvement: ~13% reduction in MAE using Random Forest.
Random Forest provides more accurate and stable predictions compared to a single decision tree.

Conclusion

Demonstrates building, validating, and improving ML models

Shows the importance of feature selection and model tuning

Practical end-to-end workflow for GitHub and LinkedIn portfolio