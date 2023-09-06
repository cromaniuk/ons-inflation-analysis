# Import libraries
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import data
ons_inflation_df = pd.read_csv('data/ons-consumer-price-inflation.csv', index_col=0)

# Data cleaning
ons_inflation_df.isna().sum()

# Create column for total
ons_inflation_total = ons_inflation_df.sum(axis=1)

# Remove 'Other' column as they contribute poorly to CPI
ons_inflation_df.drop(columns='Other', inplace=True)

X = ons_inflation_df.to_numpy()
y = ons_inflation_total.to_numpy()

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)

# Create model object
linear_model_multi_variable = LinearRegression()

# Fit model
linear_model_multi_variable.fit(train_X, train_y)

# Make Predictions on the Test Set
y_pred = linear_model_multi_variable.predict(test_X)

# Evaluate the Model
mse = mean_squared_error(test_y, y_pred)
r2 = r2_score(test_y, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Get permutation importance of each sector on CPI
perm = permutation_importance(linear_model_multi_variable, test_X, test_y, random_state=1)

sector_importances = pd.Series(perm.importances_mean, index=ons_inflation_df.columns)

fig, ax = plt.subplots()
sector_importances.plot.bar(yerr=perm.importances_std, ax=ax)
ax.set_title("Importances of each sector on CPI using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()

