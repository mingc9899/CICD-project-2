import os
import pandas as pd
import matplotlib.pyplot as plt
import skops.io as sio
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from lightgbm import LGBMRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data", "AmesHousing.csv")
MODEL_DIR = os.path.join(BASE_DIR, "Model")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

housing_df = pd.read_csv(DATA_PATH)
housing_df = housing_df.sample(frac=1, random_state=125).reset_index(drop=True)

TARGET_COL = "SalePrice"

for id_col in ["Order", "PID", "Id"]:
    if id_col in housing_df.columns:
        housing_df = housing_df.drop(id_col, axis=1)

X = housing_df.drop(TARGET_COL, axis=1)
y = housing_df[TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]

transform = ColumnTransformer(
    [
        (
            "cat_imputer_encoder",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                    ),
                ]
            ),
            cat_cols,
        ),
        (
            "num_imputer_scaler",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            num_cols,
        ),
    ]
)

pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        (
            "model",
            LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                random_state=125,
            ),
        ),
    ]
)

pipe.fit(X_train, y_train)

predictions = pipe.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse:,.2f}  MAE: {mae:,.2f}  R2: {r2:.3f}")

with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as outfile:
    outfile.write(
        f"\nRMSE = {rmse:,.2f}, MAE = {mae:,.2f}, R2 Score = {r2:.3f}."
    )

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, predictions, alpha=0.4, edgecolor="k")
lims = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
ax.plot(lims, lims, "r--", label="Perfect prediction")
ax.set_xlabel("Actual SalePrice")
ax.set_ylabel("Predicted SalePrice")
ax.set_title("Predicted vs. Actual House Prices")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "model_results.png"), dpi=120)

model_path = os.path.join(MODEL_DIR, "house_price_pipeline.skops")
sio.dump(pipe, model_path)

trusted_types = sio.get_untrusted_types(file=model_path)
print("Untrusted types found on save:", trusted_types)

reloaded_pipeline = sio.load(model_path, trusted=trusted_types)
print(reloaded_pipeline)