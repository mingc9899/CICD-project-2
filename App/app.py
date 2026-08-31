import os
import pandas as pd
import gradio as gr
import skops.io as sio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data", "AmesHousing.csv")
MODEL_PATH = os.path.join(BASE_DIR, "Model", "house_price_pipeline.skops")

trusted_types = sio.get_untrusted_types(file=MODEL_PATH)
pipe = sio.load(MODEL_PATH, trusted=trusted_types)

train_df = pd.read_csv(DATA_PATH)
for id_col in ["Order", "PID", "Id"]:
    if id_col in train_df.columns:
        train_df = train_df.drop(id_col, axis=1)
FEATURE_COLS = [c for c in train_df.columns if c != "SalePrice"]

default_row = {}
for col in FEATURE_COLS:
    if pd.api.types.is_numeric_dtype(train_df[col]):
        default_row[col] = train_df[col].median()
    else:
        default_row[col] = train_df[col].mode(dropna=True).iloc[0]

NEIGHBORHOODS = sorted(train_df["Neighborhood"].dropna().unique().tolist())
QUALITY_LABELS = ["Po", "Fa", "TA", "Gd", "Ex"]

def predict_price(
    overall_qual,
    gr_liv_area,
    total_bsmt_sf,
    garage_cars,
    year_built,
    full_bath,
    bedroom_abv_gr,
    neighborhood,
    kitchen_qual,
):
    row = default_row.copy()
    row["Overall Qual"] = overall_qual
    row["Gr Liv Area"] = gr_liv_area
    row["Total Bsmt SF"] = total_bsmt_sf
    row["Garage Cars"] = garage_cars
    row["Year Built"] = year_built
    row["Full Bath"] = full_bath
    row["Bedroom AbvGr"] = bedroom_abv_gr
    row["Neighborhood"] = neighborhood
    row["Kitchen Qual"] = kitchen_qual

    input_df = pd.DataFrame([row], columns=FEATURE_COLS)
    prediction = pipe.predict(input_df)[0]
    return f"${prediction:,.0f}"


demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(1, 10, value=int(default_row["Overall Qual"]), step=1, label="Overall Quality (1=Poor, 10=Excellent)"),
        gr.Number(value=float(default_row["Gr Liv Area"]), label="Above-Ground Living Area (sq ft)"),
        gr.Number(value=float(default_row["Total Bsmt SF"]), label="Total Basement Area (sq ft)"),
        gr.Slider(0, 4, value=int(default_row["Garage Cars"]), step=1, label="Garage Capacity (cars)"),
        gr.Number(value=int(default_row["Year Built"]), label="Year Built", precision=0),
        gr.Slider(0, 4, value=int(default_row["Full Bath"]), step=1, label="Full Bathrooms"),
        gr.Slider(0, 8, value=int(default_row["Bedroom AbvGr"]), step=1, label="Bedrooms Above Ground"),
        gr.Dropdown(NEIGHBORHOODS, value=default_row["Neighborhood"], label="Neighborhood"),
        gr.Dropdown(QUALITY_LABELS, value=default_row["Kitchen Qual"], label="Kitchen Quality"),
    ],
    outputs=gr.Textbox(label="Predicted Sale Price"),
    title="House Price Predictor",
    description=(
        "Predicts a house's sale price using a LightGBM gradient boosting "
        "model trained on the Ames Housing dataset. Fields not shown below "
        "are filled with typical (median/mode) values from the training data."
    ),
)

if __name__ == "__main__":
    demo.launch()