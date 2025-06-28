import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

######################### Create Aggregate Features  ######################################

# Transformer to create aggregate features
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        grouped = X.groupby('CustomerId')['Amount'].agg([
            ('TotalTransactionAmount', 'sum'),
            ('AverageTransactionAmount', 'mean'),
            ('TransactionCount', 'count'),
            ('StdTransactionAmount', 'std')
        ]).reset_index()
        return grouped

# Transformer to extract time-based features
class TimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'], errors='coerce')
        X['transaction_hour'] = X['TransactionStartTime'].dt.hour
        X['transaction_day'] = X['TransactionStartTime'].dt.day
        X['transaction_month'] = X['TransactionStartTime'].dt.month
        X['transaction_year'] = X['TransactionStartTime'].dt.year
        X['transaction_day_of_week'] = X['TransactionStartTime'].dt.dayofweek
        return X[['CustomerId', 'transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year', 'transaction_day_of_week']]

# Load your data
df = pd.read_csv("data/raw/data.csv")

# Define pipelines
aggregate_pipeline = Pipeline([
    ('agg_features', AggregateFeatures())
])

time_pipeline = Pipeline([
    ('time_features', TimeFeaturesExtractor())
])

# Run both pipelines
aggregate_output = aggregate_pipeline.fit_transform(df)
time_output = time_pipeline.fit_transform(df)

# Save both tables
aggregate_output.to_csv("data/processed/aggregate_features.csv", index=False)
time_output.to_csv("data/processed/time_features.csv", index=False)

######################### Encode Categorical Variables  ######################################

# Define column groups
numerical_cols = ['Amount', 'Value']
categorical_ohe = ['ChannelId', 'ProductCategory']
categorical_label = ['FraudResult']

# Define pipelines for each type
# Numerical pipeline
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical One-Hot pipeline
cat_ohe_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Categorical Label Encoding pipeline
cat_label_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('label_enc', OrdinalEncoder())
])

# Combine using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numerical_cols),
    ('cat_ohe', cat_ohe_pipeline, categorical_ohe),
    ('cat_label', cat_label_pipeline, categorical_label)
])

# Apply transformation
transformed_data = preprocessor.fit_transform(df)
transformed_df = pd.DataFrame(transformed_data)
transformed_df.to_csv("data/processed/transformed_data.csv", index=False)

######################### Handle Missing Values ######################################

# Understand missing data
print("Missing values:\n", df.isnull().sum())

# Drop columns with more than 30% missing values
threshold = 0.3
df = df[df.columns[df.isnull().mean() < threshold]]

# Drop rows with missing target (if any)
df = df.dropna(subset=['FraudResult'])

# Define column groups
numerical_cols = ['Amount', 'Value']
categorical_ohe = ['ChannelId', 'ProductCategory']
categorical_label = ['FraudResult']

# Define pipelines
# ➤ Numerical imputation: Median
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# ➤ Categorical One-Hot: Mode + OneHot
cat_ohe_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# ➤ Categorical Label: Mode + Ordinal
cat_label_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder())
])

# Combine pipelines
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numerical_cols),
    ('cat_ohe', cat_ohe_pipeline, categorical_ohe),
    ('cat_label', cat_label_pipeline, categorical_label)
])

# Transform data
transformed = preprocessor.fit_transform(df)

# Print shape and sample
print("Transformed shape:", transformed.shape)
print("First transformed row:\n", transformed[0])


######################### Normalize/Standardize  ######################################

# Define numerical columns
numerical_cols = ['Amount', 'Value']

# Choose one scaler:
use_normalization = False  # set to True to use MinMaxScaler

# Build pipeline
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler() if use_normalization else StandardScaler())
])

# Apply transformation
from sklearn.compose import make_column_transformer

preprocessor = make_column_transformer(
    (num_pipeline, numerical_cols),
    remainder='passthrough'  
)

# Fit and transform
scaled_data = preprocessor.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data)
scaled_df.to_csv('data/processed/scaled_output.csv', index=False)