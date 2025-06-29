# Tests for processing

# Uses a snippet of data for testing to see if processing.py works
import pytest
import os
import sys
import zipfile
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.data.processing import (
    process_zip,
    process_mbta,
    process_weather,
    combine_data,
    mbta_path,
    weather_path,
    processed_dir
)



# use a dummy input or a snippet of data?

@pytest.mark.order(1)
def test_process_zip_creates_mbta_data():
    raw_data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    zip_file = os.path.join(raw_data_dir, "yearly_mbta_data.zip")
    extract_to = os.path.join(raw_data_dir, "yearly_mbta_data")

    assert os.path.exists(zip_file), f"Missing: {zip_file}. Ensure it's placed in data/raw/."

    os.makedirs(extract_to, exist_ok=True)

    if not any(fname.endswith(".csv") for fname in os.listdir(extract_to)):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)


    process_zip()

    # Check output file exists
    output_file = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "mbta_data.csv")
    assert os.path.isfile(output_file), "mbta_data.csv was not created"

@pytest.mark.order(2)
def test_process_mbta_creates_processed_file():
    process_mbta(mbta_path)

    # Check output file exists
    output_file = os.path.join(processed_dir, "processed_mbta.csv")
    assert os.path.isfile(output_file), "processed_mbta.csv was not created"

@pytest.mark.order(3)
def test_process_weather_creates_processed_file():
    process_weather(weather_path)

    # Check output file exists
    output_file = os.path.join(processed_dir, "processed_weather.csv")
    assert os.path.isfile(output_file), "processed_weather.csv was not created"


@pytest.mark.order(4)
def test_process_combine_data():
    # Ensure prerequisites are run
    process_zip()
    df_mbta = process_mbta()       # must return DataFrame
    df_weather = process_weather(weather_path) # must return DataFrame

    # Now run the combination
    combine_data(df_mbta, df_weather)

    # Check if output exists
    merged_file = os.path.join(processed_dir, "merged_mbta_weather.csv")
    assert os.path.exists(merged_file), "Merged output not created"

    # Optionally validate the merged content
    df_merged = pd.read_csv(merged_file)
    assert not df_merged.empty, "Merged file is empty"
    assert "station_name" in df_merged.columns, "Missing station_name in merged data"
    assert "tavg" in df_merged.columns, "Missing tavg (temperature avg) in merged data"
