import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data.tuningModel import run_model_pipeline

def test_run_model_pipeline_outputs_csv():
    rmse = run_model_pipeline(
        csv_path="data/processed/merged_mbta_weather.csv",  
        output_csv="mbta_test_predictions.csv"
    )

    # Check file was created
    assert os.path.exists("mbta_test_predictions.csv"), "Output CSV not found."

    # Check structure of file
    df = pd.read_csv("mbta_test_predictions.csv")
    assert not df.empty, "Output CSV is empty."
    assert "actual_entries" in df.columns
    assert "predicted_entries" in df.columns

    # Check output quality
    assert rmse < 10000, f"RMSE is too high: {rmse}"
