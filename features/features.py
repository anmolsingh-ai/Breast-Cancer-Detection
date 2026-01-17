import pandas as pd
import numpy as np
from pathlib import Path
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# Load data (file path resolved relative to this file)
df = pd.read_csv("data/raw/Breast_Cancer.csv")

# Encode target
df['Status'] = df['Status'].map({'Alive': 1, 'Dead': 0})

X = df.drop('Status', axis=1)
y = df['Status']

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Preprocessing pipelines
num_pipeline = Pipeline([
	('imputer', SimpleImputer(strategy='median')),
	('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
	('imputer', SimpleImputer(strategy='most_frequent')),
	('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
	('num', num_pipeline, num_cols),
	('cat', cat_pipeline, cat_cols)
], remainder='drop')


if __name__ == '__main__':
	# Fit and transform
	X_processed = preprocessor.fit_transform(X)

	# Build feature names for resulting DataFrame
	feature_names = []
	if num_cols:
		feature_names += num_cols
	if cat_cols:
		ohe = preprocessor.named_transformers_['cat']['onehot']
		ohe_names = list(ohe.get_feature_names_out(cat_cols))
		feature_names += ohe_names

	df_processed = pd.DataFrame(X_processed, columns=feature_names)
	print('Processed shape:', df_processed.shape)
	print(df_processed.head())

	# 1. Create the directory if it doesn't exist
	output_path = Path(__file__).resolve().parent.parent / 'data' / 'processed'
	os.makedirs(output_path, exist_ok=True)

	# 2. Save the DataFrame to a CSV file
	file_name = 'processed_breast_cancer.csv'
	full_path = output_path / file_name

	df_processed.to_csv(str(full_path), index=False)

	print(f"Successfully saved preprocessed data to: {full_path}")