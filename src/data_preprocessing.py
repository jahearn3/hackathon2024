import pandas as pd
from utils import convert_cols_to_numeric

raw_path = 'data/raw/'
processed_path = 'data/processed/'
train_file = 'train.csv'
test_file = 'test.csv'

# Download the data from data/raw/train.csv
df_train = pd.read_csv(raw_path + train_file)
df_test = pd.read_csv(raw_path + test_file)

# Clean the failure column
df_train['failure'] = df_train['failure'].str.lower()
df_train['failure'] = df_train['failure'].replace({
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
df_train['failure'] = df_train['failure'].astype(int)

# Convert object columns to appropriate data types
df_train = convert_cols_to_numeric(df_train)
df_test = convert_cols_to_numeric(df_test)

# Separate binned columns from other float columns
binned_columns = []
other_float_columns = []

for col in df_train.columns:
    # Check if the column name matches the binned column naming convention
    if col[-1].isdigit() and df_train[col].dtype == 'float64': 
        if col[:-2] not in binned_columns:
            binned_columns.append(col[:-2])
    elif df_train[col].dtype == 'float64':
        other_float_columns.append(col)

# Output the lists
print("Binned Columns:", binned_columns)
print("Other Float Columns:", other_float_columns)

# Compute the correlation matrix
correlation_matrix = df_train.corr()

# Identify variables with high collinearity
y = df_train['failure']
X = df_train.drop(columns=['failure'])
threshold = 0.85
high_correlation_pairs = []
corr_matrix = X.corr()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            feature1 = corr_matrix.columns[i]
            feature2 = corr_matrix.columns[j]
            high_correlation_pairs.append((feature1, feature2, correlation_matrix.iloc[i, j]))

# Making the list of features to remove
features_to_remove = set()

# Ignoring binned features
for feature1, feature2, corr_value in high_correlation_pairs:
    if corr_value > 0.9 and not feature1[-1].isdigit() and not feature2[-1].isdigit():
        features_to_remove.add(feature2)

# Drop the features from the DataFrame
df_train.drop(columns=list(features_to_remove), inplace=True)
df_test.drop(columns=list(features_to_remove), inplace=True)

# Verify the remaining features
print("Remaining features after removing highly correlated ones:")
print(df_train.columns)

# Save the cleaned data to data/processed/
df_train.to_csv(f'{processed_path}train_processed.csv', index=False)
df_test.to_csv(f'{processed_path}test_processed.csv', index=False)
