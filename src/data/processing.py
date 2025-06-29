""" 
Pre-processing data for mbta and weather

1. Merge all data entries by their dates and station name
2. Possibly combine the data of weather and the mbta, ie. entry with how the
  weather is like on that date
  
  
1. Handle missing values or inconsistencies.
2. Align the dates in both datasets to ensure each weather record matches a 
    corresponding gate entry day. Entries for precipitation , temperature,
3. Organize stations by MBTA line, if necessary.

"""
import pandas as pd
import numpy as np
import os
import glob

# Abs Path to Data relative to script
script_dir = os.path.dirname(os.path.abspath(__file__))
mbta_path = os.path.join(script_dir, "..", '..', 'data', 'raw', 'mbta_data.csv')
weather_path = os.path.join(script_dir, "..", '..', 'data', 'raw', 'weather_data.csv')
processed_dir = os.path.join(script_dir, "..", "..", "data", "processed")
data_dir = os.path.join(script_dir, "..", '..', 'data', 'raw', 'yearly_mbta_data')


# Handling the folder of yearly mbta data

# Creates a combined csv file from a zip folder of mbta csv files
def process_zip(data_dir = data_dir):
  # Create a list of all CSV files in the directory.
  csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)

  # Read and collect all DataFrames.
  list_of_dfs = []
  for file in csv_files:
      df = pd.read_csv(file)
      list_of_dfs.append(df)

  # Concatenate all the DataFrames into one
  merged_df = pd.concat(list_of_dfs, ignore_index=True)

  output_path = os.path.join(script_dir, "..", '..', 'data', 'raw', 'mbta_data.csv')
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  merged_df.to_csv(output_path, index=False)

def process_mbta(mbta_path = mbta_path):
  # Processing MBTA data created from the combined yearly data
  df_mbta = pd.read_csv(mbta_path)
  print("Head of MBTA data: \n", df_mbta.head())
  print("MBTA date range:", df_mbta['service_date'].min(), "to", df_mbta['service_date'].max())
  print("Null values of MBTA data: \n", df_mbta.isnull().sum())

  """

    Null values in MBTA data:
      service_date          0
      time_period           0
      stop_id          115716
      station_name          0
      route_or_line         0
      gated_entries         0


    Since stop_id is just another identifier of the unique stops, 
    it is okay to drop it and just use station_name, which has no 
    missing vlaues.

    Some routes/lines also share the same station, and since our
    goal is to look at the station as a whole, it should be okay to
    drop the route_or_line column. 

    Later on, the entries with the same service_date and station_name
    will merged to get the entries number of the station for that whole day.

  """

  # Dropping stop_id and route_or_line columns
  df_mbta = df_mbta.drop(columns = ['stop_id', 'route_or_line'])
  print("Remaining columns of MBTA data: \n", df_mbta.columns)

  # Grouping by service_date and station_name
  df_mbta_grouped = df_mbta.groupby(['service_date', 'station_name'], as_index=False)['gated_entries'].sum()
  df_mbta_grouped['service_date'] = pd.to_datetime(df_mbta_grouped['service_date']).dt.date

  print(df_mbta_grouped.head())
  """
    Going through the weathers data set and finding null values
    
    Finding the important weather values and keeping them
      Then merge with the mbta dataset by the date to create a new dataset
  """

  # Creating the processed dataset for MBTA
  mbta_processed_file = os.path.join(processed_dir, "processed_mbta.csv")

  os.makedirs(processed_dir, exist_ok=True)
  df_mbta_grouped.to_csv(mbta_processed_file, index=False)
  return df_mbta_grouped


def process_weather(weather_path):
  # Processing Weather data
  df_weather = pd.read_csv(weather_path)
  print("Head of weather data: \n", df_weather.head())
  print("Null values of weather data: \n", df_weather.isnull().sum())

  """ 

  Data Columns:
    time: The date in string format
    tavg: The average air temperature in Celsius, as a float
    tmin: The minimum air temperature in Celsius, as a float
    tmax: The maximum air temperature in Celsius, as a float
    prcp: The daily precipitation total in millimeters, as a float
    wdir: The average wind direction in degrees, as a float (Not as important)
    wspd: The average wind speed in kilometers per hour, as a float
    pres: The average sea-level air pressure in hectopascals, as a float (Not as important)

  """

  """

    Null values in weather data:
      time      0
      tavg      1
      tmin      0
      tmax      0
      prcp      0
      wdir    587
      wspd      0
      pres    161
      
  """

  # Inputing the missing null value for tavg
  # Uses the avg of tmax and tmin for that day
  df_weather.loc[df_weather['tavg'].isnull(), 'tavg'] = (df_weather['tmin'] + df_weather['tmax']) / 2

  print("Null values of weather data: \n", df_weather.isnull().sum())

  # Dropping wdir and pres columns
  df_weather = df_weather.drop(columns = ['wdir', 'pres'])
  print("Remaining columns of weather data: \n", df_weather.columns)

  # Renaming time in mbta data to service_date
  df_weather.rename(columns={'time': 'service_date'}, inplace=True)
  df_weather['service_date'] = pd.to_datetime(df_weather['service_date']).dt.date

  # Creating Weather processed file
  weather_processed_file = os.path.join(processed_dir, "processed_weather.csv")
  os.makedirs(processed_dir, exist_ok=True)
  df_weather.to_csv(weather_processed_file, index=False)
  return df_weather
  
  
  
# Combines cleaned mbta and weather data
def combine_data(df_mbta_grouped, df_weather):
  # Merge MBTA and weather data on the date)
  df_merged = pd.merge(df_mbta_grouped, df_weather, on='service_date', how='inner')

  # Weather data only goes up to March 1st, 2023, so the merged data should also only go up to that date
  end_date = pd.to_datetime("2023-03-01")
  df_merged['service_date'] = pd.to_datetime(df_merged['service_date'])
  df_merged = df_merged[df_merged['service_date'] <= end_date]
  print("Merged DataFrame head: \n", df_merged.head())



  # Creating processed csv for merged mbta and weather
  merged_processed_file = os.path.join(processed_dir, "merged_mbta_weather.csv")
  os.makedirs(processed_dir, exist_ok=True)
  df_merged.to_csv(merged_processed_file, index=False)
    