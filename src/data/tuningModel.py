import pandas as pd
import numpy as np
import holidays
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import os


def assign_line_colors(df, line_colors):
    df['line_color'] = 'Other'
    for color, stations in line_colors.items():
        for station in stations:
            mask = df['station_name'].str.contains(station, case=False, na=False)
            df.loc[mask, 'line_color'] = color
    return df


def run_model_pipeline(
    csv_path='../../data/processed/merged_mbta_weather.csv',
    output_csv='mbta_test_predictions.csv',
    plot=False
):
    # Load data
    df = pd.read_csv(csv_path)
    df['service_date'] = pd.to_datetime(df['service_date'])

    # Feature engineering
    df['day_of_week'] = df['service_date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    def month_to_season(month):
        if month in [12, 1, 2]: return 'winter'
        elif month in [3, 4, 5]: return 'spring'
        elif month in [6, 7, 8]: return 'summer'
        else: return 'fall'

    df['month'] = df['service_date'].dt.month
    df['season'] = df['month'].apply(month_to_season)
    df = pd.get_dummies(df, columns=['season'], drop_first=True)

    years = df['service_date'].dt.year.unique().tolist()
    us_hols = holidays.US(years=years)
    df['is_holiday'] = df['service_date'].apply(lambda d: 1 if d in us_hols else 0)

    holiday_dates = sorted(us_hols.keys())

    def days_to_next_hol(dt):
        d = dt.date()
        future = [(hol - d).days for hol in holiday_dates if hol >= d]
        return min(future) if future else np.nan

    def days_from_prev_hol(dt):
        d = dt.date()
        past = [(d - hol).days for hol in holiday_dates if hol <= d]
        return min(past) if past else np.nan

    df['days_to_next_hol']   = df['service_date'].apply(days_to_next_hol)
    df['days_from_prev_hol'] = df['service_date'].apply(days_from_prev_hol)

    df['week_start'] = df['service_date'] - pd.to_timedelta(df['service_date'].dt.weekday, unit='d')
    df['lag1']  = df['gated_entries'].shift(1)
    df['roll7'] = df['gated_entries'].shift(1).rolling(7).mean()
    df = df.dropna()

    df['month_norm'] = 2 * np.pi * (df['month'] - 1) / 12
    df['month_sin']  = np.sin(df['month_norm'])
    df['month_cos']  = np.cos(df['month_norm'])

    df['is_covid_period'] = ((df['service_date'] >= '2020-03-15') & (df['service_date'] < '2021-03-01')).astype(int)
    df['is_recovery_period'] = ((df['service_date'] >= '2021-03-01') & (df['service_date'] < '2022-01-01')).astype(int)
    df['is_post_covid'] = (df['service_date'] >= '2022-01-01').astype(int)
    df['covid_weekend'] = df['is_covid_period'] * df['is_weekend']
    df['recovery_weekend'] = df['is_recovery_period'] * df['is_weekend']

    line_colors = {
        'Red': ['Alewife', 'Davis', 'Porter', 'Harvard', 'Central', 'Kendall', 'Charles/MGH', 'Park Street', 
            'Downtown Crossing', 'South Station', 'Broadway', 'Andrew', 'JFK/UMass', 'Savin Hill',
            'Fields Corner', 'Shawmut', 'Ashmont', 'North Quincy', 'Wollaston', 'Quincy Center', 
            'Quincy Adams', 'Braintree'],
        'Green': ['Lechmere', 'Science Park', 'North Station', 'Haymarket', 'Government Center', 
              'Park Street', 'Boylston', 'Arlington', 'Copley', 'Hynes', 'Kenmore', 'Prudential',
              'Symphony', 'Northeastern', 'Museum of Fine Arts', 'Longwood Medical Area', 'Brigham Circle',
              'Fenwood Road', 'Mission Park', 'Riverway', 'Back of the Hill', 'Heath Street', 'Cleveland Circle',
              'Beaconsfield', 'Reservoir', 'Chestnut Hill', 'Newton Centre', 'Boston College'],
        'Orange': ['Oak Grove', 'Malden Center', 'Wellington', 'Assembly', 'Sullivan Square', 'Community College',
              'North Station', 'Haymarket', 'State', 'Downtown Crossing', 'Chinatown', 'Tufts Medical Center',
              'Back Bay', 'Massachusetts Avenue', 'Ruggles', 'Roxbury Crossing', 'Jackson Square',
              'Stony Brook', 'Green Street', 'Forest Hills'],
        'Blue': ['Wonderland', 'Revere Beach', 'Beachmont', 'Suffolk Downs', 'Orient Heights', 'Wood Island',
            'Airport', 'Maverick', 'Aquarium', 'State', 'Government Center', 'Bowdoin'],
        'Silver': ['South Station', 'Courthouse', 'World Trade Center']
    }
    df = assign_line_colors(df, line_colors)

    train_mask = df['service_date'] <= '2022-03-01'
    X_cols = [
        'tavg', 'tmin', 'tmax', 'prcp', 'wspd',
        'is_weekend', 'is_holiday', 'days_to_next_hol', 'days_from_prev_hol',
        'season_spring', 'season_summer', 'season_winter',
        'lag1', 'roll7',
        'month_sin', 'month_cos',
        'is_covid_period', 'is_recovery_period', 'is_post_covid',
        'covid_weekend', 'recovery_weekend']

    df_enc = pd.get_dummies(df, columns=['station_name'], drop_first=True)
    station_dummies = [c for c in df_enc.columns if c.startswith('station_name_')]
    X_cols_ext = X_cols + station_dummies

    X = df_enc[X_cols_ext]
    y = df_enc['gated_entries']
    train_mask = df_enc['service_date'] <= '2022-03-01'

    X_train, X_test = X.loc[train_mask], X.loc[~train_mask]
    y_train, y_test = y.loc[train_mask], y.loc[~train_mask]

    pipeline = Pipeline([
        ('poly',  PolynomialFeatures(degree=2, include_bias=False)),
        ('scale', StandardScaler()),
        ('ridge', RidgeCV(alphas=[0.1, 1.0, 10.0], cv=TimeSeriesSplit(n_splits=5)))
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_int = np.rint(y_pred).astype(int)
    y_pred_clipped = np.clip(y_pred_int, 0, None)

    df_enc.loc[~train_mask, 'predicted_entries'] = y_pred_clipped

    start_date = '2022-03-02'
    end_date   = '2023-03-01'
    test_mask_orig = (
        (df['service_date'] >= start_date) &
        (df['service_date'] <=   end_date)
    )

    test_orig = df.loc[test_mask_orig, [
        'service_date','station_name','tavg','prcp','wspd','gated_entries'
    ]].copy()
    test_orig.rename(columns={'gated_entries':'actual_entries'}, inplace=True)
    test_orig['predicted_entries'] = df_enc.loc[test_mask_orig, 'predicted_entries'].values
    test_orig.to_csv(output_csv, index=False)

    rmse = np.sqrt(mean_squared_error(test_orig['actual_entries'], test_orig['predicted_entries']))
    return rmse
