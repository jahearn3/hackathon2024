# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
path = '../data/raw/train.csv'
# Download the data from data/raw/train.csv
df = pd.read_csv(path)

# %%
# Look at value_counts of failure column
# df['failure'].value_counts()

# %%
# Clean the failure column
df['failure'] = df['failure'].str.lower()
df['failure'] = df['failure'].replace({
    'n': 0,
    'no': 0,
    'false': 0,
    False: 0,
    '0': 0,
    'y': 1,
    'yes': 1,
    'true': 1,
    '1': 1,
    True: 1
})
# <ipython-input-20-deb4bb0de4ba>:4: FutureWarning: 
# Downcasting behavior in `replace` is deprecated and will be removed 
# in a future version. To retain the old behavior, explicitly call 
# `result.infer_objects(copy=False)`. To opt-in to the future behavior, 
# set `pd.set_option('future.no_silent_downcasting', True)`

df['failure'] = df['failure'].astype(int)

# Check dtype of failure column
# df['failure'].dtype

# %%
# General Information

# Get a summary of the dataset
# print(df.info())

# Check for missing values
for col in df.columns:
    nulls = df[col].isnull().sum()
    if nulls > 0:
        print(f'{col} has {nulls} missing value(s)')    

# %%
# Convert object columns to appropriate data types

# For each column, convert nulls to np.nan, and convert to float if possible
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].replace({"": np.nan, "nan": np.nan, "NAN": np.nan, "NaN": np.nan, "na": np.nan})
        if df[col].str.isnumeric().all():
            df[col] = df[col].astype(float)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

# %%
# Descriptive Statistics
# Get descriptive statistics
print(df.describe())

# %%
# Separate binned columns from other float columns
binned_columns = []
other_float_columns = []

for col in df.columns:
    # Check if the column name matches the binned column naming convention
    if col[-1].isdigit() and df[col].dtype == 'float64': 
        if col[:-2] not in binned_columns:
            binned_columns.append(col[:-2])
    elif df[col].dtype == 'float64':
        other_float_columns.append(col)

# Output the lists
print("Binned Columns:", binned_columns)
print("Other Float Columns:", other_float_columns)

# %%
# Expore binned column data
# Put 10 histograms on the same plot
# num_bins = 10
# for col in binned_columns:
#     col_list = [col + ' ' + str(i) for i in range(num_bins)]
#     if col == 'Crested Butte':
#         for c in col_list:
#             # Plot boxplot
#             plt.figure(figsize=(10, 5))
#             sns.boxplot(data=df, x=c, y='failure')
#             plt.title(f'Boxplot for {c}')
#             plt.xlabel(c)
#             plt.ylabel('Failure')
#             # Decrease tick labels
#             plt.xticks(rotation=45)
#             plt.show()
    # plt.figure(figsize=(10, 15))
    # long_data = pd.melt(df, id_vars='failure', value_vars=col_list,
    #                     var_name='Bin', value_name='Value')
    # sns.boxplot(data=long_data, x='Bin', y='failure', hue='Value')  # Replace with your target variable
    # plt.title(f'Boxplot for {col}')
    # plt.xlabel('Bins')
    # plt.ylabel('Failure')
    # # for i in range(num_bins):
    # #     column = col + ' ' + str(i)
    # #     # print(df[column].describe())
    # #     plt.subplot(num_bins, 1, i + 1)
    # #     sns.boxplot(data=df, x=col, y='failure') 
    #     # sns.histplot(df[df[column]], bins=num_bins, kde=False)
    #     # plt.ylabel(f'Frequency in bin {i}')
    #     # plt.ylim(0, df[df[column]].shape[0] // 10 + 1)
    #     # print(f"Unique values in {column}: {df[column].unique()}")
    #     # print(f"Crosstab for {column} and failure: {pd.crosstab(df[column], df['failure'])}")
    # # plt.figure(figsize=(10, 5))
    # # sns.countplot(data=df, x=col_list)
    # # plt.title(f'Histogram for {col}')
    # # plt.xlabel(col)
    # plt.tight_layout()
    # plt.show()

# %%
# Data Distribution
# Plot distribution of numerical features
# numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
for feature in other_float_columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[feature], bins=30, kde=False)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# %%
# Categorical Features - There are no categorical features in this dataset
# Plot counts of categorical features
# categorical_features = df.select_dtypes(include=[object]).columns.tolist()
# for feature in categorical_features:
#     plt.figure(figsize=(10, 5))
#     sns.countplot(data=train_data, x=feature)
#     plt.title(f'Count of {feature}')
#     plt.xticks(rotation=45)
#     plt.show()

# %%
# Correlation Matrix
# Compute the correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# %% 
# Print out columns with high correlation with 'failure'
threshold = 0.5
for col in correlation_matrix.columns:
    if abs(correlation_matrix[col]['failure']) > threshold:
        print(f"{col}: {correlation_matrix[col]['failure']}")


# %%
# Pairplot
# Pairplot for a subset of features (for large datasets, this might be slow)
# sns.pairplot(df[numerical_features[:5]])  # Adjust the number of features as needed
# plt.show()

# %%
# Target Variable Analysis
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='failure')
plt.title('Count of Failure')
plt.show()

# %%
# Calculate percentage of failures
total_failures = df['failure'].sum()
total_samples = df.shape[0]
failure_percentage = total_failures / total_samples * 100
print(f'Percentage of Failures: {failure_percentage:.2f}%')

# %%
# Cost of No Inspections
print(f'Cost of No Inspections: ${(total_failures * 750):,}-${(total_failures * 2000):,}')

# %% 
# Cost of Inspecting Every Truck
inspection_cost_min = 100 * total_samples
inspection_cost_max = 300 * total_samples
total_non_failures = total_samples - total_failures
voucher_cost = 250 * total_non_failures
repair_cost = 150 * total_failures
inspect_every_truck_min = inspection_cost_min + repair_cost + voucher_cost
inspect_every_truck_max = inspection_cost_max + repair_cost + voucher_cost
print(f'Cost of Inspecting Every Truck: ${inspect_every_truck_min:,}-${inspect_every_truck_max:,}')

# %%
# Cost of Only Inspecting Trucks with a Predicted Failure (Optimal Solution)
inspection_cost_min = 100 * total_failures
inspection_cost_max = 300 * total_failures
repair_cost = 150 * total_failures
perfect_inspection_min = inspection_cost_min + repair_cost
perfect_inspection_max = inspection_cost_max + repair_cost
print(f'Cost of Only Inspecting Trucks with a Predicted Failure: ${perfect_inspection_min:,}-${perfect_inspection_max:,}')

# %%
# Cost per truck
inspect_every_truck_max /= total_samples
inspect_every_truck_min /= total_samples
perfect_inspection_min /= total_samples
perfect_inspection_max /= total_samples
total_failures /= total_samples


# %%
# Plot the cost of different strategies
strategies = {
    'Inspect Every Truck': {'min': inspect_every_truck_min, 'max': inspect_every_truck_max},
    'No Inspections': {'min':  total_failures * 750, 'max': total_failures * 2000},
    'Perfectly Selected Inspections': {'min': perfect_inspection_min, 'max': perfect_inspection_max},
    'ML Model': {'min': 9.5, 'max': 20.42}
}
# Initialize the figure
plt.figure(figsize=(10, 4))
# Create the plot
# For each element in strategy, create a box from its minimum to its maximum value
i = 0
# Use these x values for cost per truck
# x = [440, 21, 5.6]
x = [440, 29, 8, 14]
# Use these x values for whole price, not price per truck
# x = [25.0, 1.17, 0.335]
# y = [-0.1, 0.9, 1.9]
# y = [0.5, 1.0, 1.5]
# y = [0., 0., 1.]
# Use this loop for cost per truck
for k, v in strategies.items():
    plt.plot([v['min'], v['max']], [i, i],  marker='|', markersize=10)
    text = f'{k}:\n\${v["min"]:.2f}-\${v["max"]:.2f}'
    # plt.text(x[i], y[i], text, horizontalalignment='center')
    plt.text(x[i], i + 0.5, text, horizontalalignment='center')
    i += 1
# Use the below loop when doing the whole price, not the price per truck
# for k, v in strategies.items():
#     plt.plot([v['min'] / 1.E+06, v['max'] / 1.E+06], [i, i],  marker='|', markersize=10)
#     text = f'{k}:\n\${round(v["min"] / 1.E+06, 2):,}M-\${round(v["max"] / 1.E+06, 2):,}M'
#     plt.text(x[i], y[i], text, horizontalalignment='center')
#     i += 1
# plt.xticks(range(i), strategies.keys())
# plt.title('Cost of Different Inspection Strategies')
# plt.xlabel('Cost')
plt.xlabel('Cost per Truck')
# Make x axis log
plt.xscale('log')
# Format x tick labels in terms of dollars
plt.gca().xaxis.set_major_formatter('${:,.0f}'.format)
# Flip y axis
plt.gca().invert_yaxis()
# Hide y tick labels
plt.yticks([])
# Put x axis on top
plt.gca().xaxis.tick_top()
# Put x axis label on top
plt.gca().xaxis.set_label_position('top')
# Hide top and right spines
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.show()

# %%
# Identify variables with high collinearity
y = df['failure']
X = df.drop(columns=['failure'])
threshold = 0.85
high_correlation_pairs = []
corr_matrix = X.corr()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            feature1 = corr_matrix.columns[i]
            feature2 = corr_matrix.columns[j]
            high_correlation_pairs.append((feature1, feature2, correlation_matrix.iloc[i, j]))

# %%
# Print the pairs of highly correlated features
if high_correlation_pairs:
    print("Highly correlated feature pairs (with correlation coefficient):")
    for feature1, feature2, corr_value in high_correlation_pairs:
        if corr_value > 0.9:
            print(f"{feature1} and {feature2}: {corr_value:.2f}")
else:
    print("No highly correlated features found.")

# %%
# Making the list of features to remove
features_to_remove = set()

# Ignoring binned features
for feature1, feature2, corr_value in high_correlation_pairs:
    if corr_value > 0.9 and feature1[-1].isdigit() == False and feature2[-1].isdigit() == False:
        features_to_remove.add(feature2)

# %%
# Drop the features from the DataFrame
df.drop(columns=list(features_to_remove), inplace=True)

# %%
# Verify the remaining features
print("Remaining features after removing highly correlated ones:")
print(df.columns)

# %%
# Split into train and test sets
from sklearn.model_selection import train_test_split

# Split the data into train and test sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=3)

# %%
# Scale the data
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data
scaler.fit(train_data.drop(columns='failure'))

# Transform the training and test data
train_data_scaled = scaler.transform(train_data.drop(columns='failure'))
test_data_scaled = scaler.transform(test_data.drop(columns='failure'))

# %%
# Convert the scaled data back to a DataFrame
train_data_scaled = pd.DataFrame(train_data_scaled, columns=train_data.drop(columns='failure').columns)
test_data_scaled = pd.DataFrame(test_data_scaled, columns=test_data.drop(columns='failure').columns)

# %%
# Run a simple model
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Initialize the model
model = HistGradientBoostingClassifier()

# Fit the model
model.fit(train_data_scaled, train_data['failure'])

# Predict on the test data
predictions = model.predict(test_data_scaled)

# Calculate the accuracy
accuracy = accuracy_score(test_data['failure'], predictions)
print(f'Accuracy: {accuracy:.2f}')

# %%
# Print confusion matrix
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
conf_matrix = confusion_matrix(test_data['failure'], predictions)

# Print the confusion matrix
print('Confusion Matrix:')
print(conf_matrix)

# %%
# Identify the true positives, false positives, true negatives, and false negatives
true_positives = conf_matrix[1, 1]
false_positives = conf_matrix[0, 1]
true_negatives = conf_matrix[0, 0]
false_negatives = conf_matrix[1, 0]

# %%
# Calculate cost of the model
inspection_cost_min = 100 * (true_positives + false_positives)
inspection_cost_max = 300 * (true_positives + false_positives)
voucher_cost = 250 * false_positives
repair_cost = 150 * true_positives
failed_repair_cost_min = 750 * false_negatives
failed_repair_cost_max = 2000 * false_negatives
total_cost_min = inspection_cost_min + voucher_cost + repair_cost + failed_repair_cost_min
total_cost_max = inspection_cost_max + voucher_cost + repair_cost + failed_repair_cost_max
print(f'Cost of the model: ${total_cost_min:,}-${total_cost_max:,}')
# %%
# Cost per truck
total_trucks = test_data.shape[0]
cost_per_truck_min = total_cost_min / total_trucks
cost_per_truck_max = total_cost_max / total_trucks
print(f'Cost per truck: ${cost_per_truck_min:.2f}-${cost_per_truck_max:.2f}')
# %%
# Calculate precision, specificity, recall
# We want to focus on recall
precision = true_positives / (true_positives + false_positives)
specificity = true_negatives / (true_negatives + false_positives)
recall = true_positives / (true_positives + false_negatives)
print(f'Precision: {precision:.4f}')
print(f'Specificity: {specificity:.4f}')
print(f'Recall: {recall:.4f}')
# %%
