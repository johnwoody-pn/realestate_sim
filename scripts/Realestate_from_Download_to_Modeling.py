#realestate_from_download_to_modeling

import os
import pandas as pd
import numpy as np
import kagglehub
import shutil
import logging
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import joblib


###
# step 0: Log 
###
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "gridsearch_log.txt"),
    filemode='w',
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

###
# Step 1: Download the data
###
print("Step 1: Downloading the data")
dataset_path= kagglehub.dataset_download("yuliiabulana/canada-housing")

raw_data_dir= "data/raw"
os.makedirs(raw_data_dir, exist_ok=True)
for file_name in os.listdir(dataset_path):
    src= os.path.join(dataset_path, file_name)
    dst= os.path.join(raw_data_dir, file_name)
    shutil.move(src, dst)

# we only use the cleaned_canada.csv for the main analysis
csv_path= os.path.join(raw_data_dir, "cleaned_canada.csv")
df= pd.read_csv(csv_path)
print("Dataset loaded:", df.shape)

###
# Step 2: Clean the data
###
print("Step 2: Cleaning the data")
df= df.replace(["", " ", "None", "none", "null", "Null", "NULL"], np.nan)

def remove_outlier_IQR(df, group_col, target_cols):
    result = pd.DataFrame()

    for ptype in df[group_col].unique():
        sub_df = df[df[group_col] == ptype].copy()
        
        for col in target_cols:
            Q1 = sub_df[col].quantile(0.25)
            Q3 = sub_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            sub_df = sub_df[(sub_df[col] >= lower_bound) & (sub_df[col] <= upper_bound)]
        
        result = pd.concat([result, sub_df], ignore_index=True)
    
    return result

df = remove_outlier_IQR(df, group_col='Property Type', target_cols=['Price', 'Square Footage'])

###
# Step 3:Binary and categorical columns
###
binary_cols= ['Garage', 'Parking', 'Fireplace', 'Waterfront', 'Pool', 'Garden', 'Balcony']
for col in binary_cols:
    df[col]= df[col].map({"Yes": 1, "No": 0})

cat_cols = ['City', 'Province', 'Property Type', 'Basement', 'Exterior',
    'Heating', 'Flooring', 'Roof', 'Sewer']
for col in cat_cols:
    df[col] = df[col].astype('category')


###
# Step 4. Modeling with CV
###
print("Building XGboost model with Cross-Validation")

target = 'Price'
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

model = XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    enable_categorical=True,
    random_state=42
)

param_grid = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 200],
    "subsample": [0.8],
    "colsample_bytree": [0.8]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, param_grid, scoring="r2", cv=cv, verbose=1, n_jobs=-1)
grid_search.fit(X, y)

###
# Step 5. Save the model
###

best_model = grid_search.best_estimator_
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model_xgb.pkl")

print("Best parameters:", grid_search.best_params_)
print("Best R2 score:", grid_search.best_score_)

logging.info("Best papameters: %s", grid_search.best_params_)
logging.info("Best R2score: %s", grid_search.best_score_)

os.makedirs("results", exist_ok=True)
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_csv("results/cv_results.csv", index=False)
