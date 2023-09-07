## Import libraries
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

## Import CPIH data
ons_inflation_all = pd.read_csv('data/ons-consumer-price-inflation.csv', index_col=0)

## Data cleaning
ons_inflation_all.isna().sum()

# Split dataframes into Total CPIH and contributing sectors
ons_inflation_all['Total'] = ons_inflation_all.sum(axis=1)
ons_inflation_df = ons_inflation_all[['Food and non-alcoholic beverages',
'Alcohol and tobacco', 'Housing and household services', 'Transport', 'Other']]
ons_inflation_total = ons_inflation_all[['Total']]
ons_inflation_all = pd.concat([ons_inflation_df, ons_inflation_total], axis=1)

# Create plot of all sectors and CPIH against time
plt.plot(ons_inflation_all.index, ons_inflation_all[['Food and non-alcoholic beverages',
'Alcohol and tobacco', 'Housing and household services', 'Transport', 'Other', 'Total']])

# Add title and axis labels
plt.title('Contribution to CPIH of different sectors', fontname="Arial", size=16)
plt.xlabel('Date')
plt.ylabel('Contribution to CPIH')
plt.legend(['Food and non-alcoholic beverages',
'Alcohol and tobacco', 'Housing and household services', 'Transport', 'Other', 'Total CPIH'], loc="upper right")
ax = plt.gca()
plt.xticks(rotation=90)
ax.xaxis.set_major_locator(plt.MaxNLocator(30))

## Linear regression

# Remove 'Other' column as they contribute poorly to CPIH
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

# Get permutation importance of each sector on CPIH
perm = permutation_importance(linear_model_multi_variable, test_X, test_y, random_state=1)

sector_importances = pd.Series(perm.importances_mean, index=ons_inflation_df.columns)

fig, ax = plt.subplots()
sector_importances.plot.bar(yerr=perm.importances_std, ax=ax)
ax.set_title("Importances of each sector on CPIH using permutation on full model", fontname="Arial", size=16)
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()

## Import housing sector breakdown data
housing_breakdown_df = pd.read_csv('data/ons-housing-and-household-services-contribution.csv')

# Create stacked bar chart
housing_breakdown_df.plot(x='Date', kind='bar', stacked=True,
        title='Contributions of housing components to the annual CPIH inflation rate, UK, January 2015 to July 2023')
plt.show()
