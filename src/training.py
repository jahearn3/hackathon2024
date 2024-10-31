# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, recall_score, classification_report, make_scorer
from utils import custom_scorer

# %%
processed_path = '../data/processed/'
# processed_path = 'data/processed/'
# %%
# Load the cleaned data
print('Loading data...')
df = pd.read_csv(f'{processed_path}train_processed.csv')
df_test = pd.read_csv(f'{processed_path}test_processed.csv')
# %%
# Split the data into train and test sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=3)
# %%
# Scale the data
print('Scaling data...')
scaler = StandardScaler()

# Fit the scaler on the training data
scaler.fit(train_data.drop(columns='failure'))

# Transform the training and test data
train_data_scaled = scaler.transform(train_data.drop(columns='failure'))
test_data_scaled = scaler.transform(test_data.drop(columns='failure'))
df_test_scaled = scaler.transform(df_test)

# Convert the scaled data back to a DataFrame
train_data_scaled = pd.DataFrame(train_data_scaled, columns=train_data.drop(columns='failure').columns)
test_data_scaled = pd.DataFrame(test_data_scaled, columns=test_data.drop(columns='failure').columns)
df_test_scaled = pd.DataFrame(df_test_scaled, columns=df_test.columns)

# %%
# Grid Search on Full Model
param_grid = {
    'max_iter': [500, 600],  # [100, 200],
    'max_leaf_nodes': [77, 99],  # [31, 63],
    'min_samples_leaf': [8, 10, 12],  # [20, 30],
    # 'learning_rate': [0.1, 0.01]
}

print('Beginning grid search...')
custom_scorer_func = make_scorer(custom_scorer, greater_is_better=False)
grid_search = GridSearchCV(HistGradientBoostingClassifier(class_weight='balanced'), param_grid, scoring=custom_scorer_func, cv=5)
grid_search.fit(train_data_scaled, train_data['failure'])
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# %%
# Predictions on test set
# y_pred = best_model.predict(test_data_scaled)

# # %%
# # Calculate recall
# recall = recall_score(test_data['failure'], y_pred)
# print("Recall on Test Set:", recall)
# print(classification_report(test_data['failure'], y_pred))

# %%
# Grid Search on Binned Model and Other Model

# Get lists of binned columns and other columns
# Separate binned columns from other float columns
binned_columns = []
other_float_columns = []

for col in train_data_scaled.columns:
    # Check if the column name matches the binned column naming convention
    if col[-1].isdigit() and train_data_scaled[col].dtype == 'float64': 
        binned_columns.append(col)
    elif train_data_scaled[col].dtype == 'float64':
        other_float_columns.append(col)

# Output the lists
# print("Binned Columns:", binned_columns)
# print("Other Float Columns:", other_float_columns)

# Save binned and other dataframes
train_data_binned = train_data_scaled[binned_columns]
train_data_other = train_data_scaled[other_float_columns]

# Initialize binned and other model grid searches
binned_model_grid_search = GridSearchCV(HistGradientBoostingClassifier(class_weight='balanced'), param_grid, scoring=custom_scorer_func, cv=5)
other_model_grid_search = GridSearchCV(HistGradientBoostingClassifier(class_weight='balanced'), param_grid, scoring=custom_scorer_func, cv=5)

# Fit the grid searches
binned_model_grid_search.fit(train_data_binned, train_data['failure'])
other_model_grid_search.fit(train_data_other, train_data['failure'])

# Get the best models
binned_best_model = binned_model_grid_search.best_estimator_
other_best_model = other_model_grid_search.best_estimator_

# Get the best parameters
binned_best_params = binned_model_grid_search.best_params_
other_best_params = other_model_grid_search.best_params_

# Print the best parameters
print("Binned Best Parameters:", binned_best_params)
print("Other Best Parameters:", other_best_params)


# %%
# Build models with the best params found in grid searches
model = HistGradientBoostingClassifier(**best_params, class_weight='balanced')
binned_model = HistGradientBoostingClassifier(**binned_best_params, class_weight='balanced')
other_model = HistGradientBoostingClassifier(**other_best_params, class_weight='balanced')

# Fit the models
model.fit(train_data_scaled, train_data['failure'])
binned_model.fit(train_data_binned, train_data['failure'])
other_model.fit(train_data_other, train_data['failure'])

# %%
# Predict on the test data
probabilities = model.predict_proba(test_data_scaled)[:, 1]
threshold = 0.75  # Example threshold was 0.3
predictions = (probabilities >= threshold).astype(int)

test_data_binned = test_data_scaled[binned_columns]
probabilities_binned = binned_model.predict_proba(test_data_binned)[:, 1]
predictions_binned = (probabilities_binned >= threshold).astype(int)

test_data_other = test_data_scaled[other_float_columns]
probabilities_other = other_model.predict_proba(test_data_other)[:, 1]
predictions_other = (probabilities_other >= threshold).astype(int)

# %% 
# Build the voting classifier and predict on the eval data
voting_clf = VotingClassifier(estimators=[
    ('model', model),
    ('binned_model', binned_model),
    ('other_model', other_model)
], voting='soft')

voting_clf.fit(train_data_scaled, train_data['failure'])

probabilities_voting = voting_clf.predict_proba(test_data_scaled)[:, 1]
predictions_voting = (probabilities_voting >= threshold).astype(int)

# %%
# Calculate the confusion matrix
conf_matrix = confusion_matrix(test_data['failure'], predictions_voting)
print('Confusion Matrix:')
print(conf_matrix)
# %%
# Identify the true positives, false positives, true negatives, and false negatives
true_positives = conf_matrix[1, 1]
false_positives = conf_matrix[0, 1]
true_negatives = conf_matrix[0, 0]
false_negatives = conf_matrix[1, 0]
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
# Retrain the model on all the training data

# Scale the full training dataset
scaler.fit(df.drop(columns='failure'))
df_scaled = scaler.transform(df.drop(columns='failure'))
df_scaled = pd.DataFrame(df_scaled, columns=df.drop(columns='failure').columns)

# %%
# Build the full models using the best params
full_model = HistGradientBoostingClassifier(**best_params, class_weight='balanced')
full_model.fit(df_scaled, df['failure'])

full_binned_model = HistGradientBoostingClassifier(**binned_best_params, class_weight='balanced')
full_binned_model.fit(df_scaled[binned_columns], df['failure'])

full_other_model = HistGradientBoostingClassifier(**other_best_params, class_weight='balanced')
full_other_model.fit(df_scaled[other_float_columns], df['failure'])

# %% 
# Build the full voting classifier
full_voting_model = VotingClassifier(estimators=[
    ('full_model', full_model),
    ('full_binned_model', full_binned_model),
    ('full_other_model', full_other_model)
], voting='soft')

full_voting_model.fit(df_scaled, df['failure'])

# %%
# Make predictions on the test set
probabilities_test = full_voting_model.predict_proba(df_test_scaled)[:, 1]
predictions_test = (probabilities_test >= threshold).astype(int)

# %%
# Save the predictions to a CSV file
df_test['predicted_failure'] = predictions_test
# Drop the other columns
df_test = df_test[['predicted_failure']]
df_test.to_csv('../data/predictions.csv', index=True)

# %%
# Compare data/predictions.csv with data/predictions_partial.csv
pred_partial = pd.read_csv('../data/predictions_partial.csv')
pred_full = pd.read_csv('../data/predictions_full.csv')
pred_voting = pd.read_csv('../data/predictions.csv')

# Merge the two DataFrames
merged = pred_partial.merge(pred_full, left_index=True, right_index=True, suffixes=('_partial', '_full'))
merged = merged.merge(pred_voting, left_index=True, right_index=True)
# Calculate percentage of rows where the predictions differ
diff = (merged['predicted_failure_partial'] != merged['predicted_failure_full']).mean()
print(diff)

# Calculate percentage of rows where the predictions are the same
same = (merged['predicted_failure_partial'] == merged['predicted_failure_full']).mean()
print(same)

diff = (merged['predicted_failure_partial'] != merged['predicted_failure']).mean()
print(diff)

# Calculate percentage of rows where the predictions are the same
same = (merged['predicted_failure_partial'] == merged['predicted_failure']).mean()
print(same)

diff = (merged['predicted_failure'] != merged['predicted_failure_full']).mean()
print(diff)

# Calculate percentage of rows where the predictions are the same
same = (merged['predicted_failure'] == merged['predicted_failure_full']).mean()
print(same)

# %%
# Count number of positive predictions
print('Partial Model Predictions:')
print(pred_partial['predicted_failure'].value_counts())
print('Full Model Predictions:')
print(pred_full['predicted_failure'].value_counts())
print('Voting Model Predictions:')
print(pred_voting['predicted_failure'].value_counts())
# %%
