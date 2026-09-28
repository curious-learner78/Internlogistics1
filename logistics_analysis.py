"""
Logistics Data Analysis & Route Optimization Pipeline
Task: Week 1 Strategic Planning and Data Exploration
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def preprocess_logistics_data(filepath: str) -> pd.DataFrame:
  """Loads, cleans, and engineers features for the logistics dataset."""
  df = pd.read_csv(filepath)

  # Drop missing coordinates or timestamp records
  df.dropna(
      subset=['delivery_lat', 'delivery_long', 'dispatch_time'], inplace=True
  )

  # Datetime conversions
  df['dispatch_time'] = pd.to_datetime(df['dispatch_time'])
  df['delivery_time'] = pd.to_datetime(df['delivery_time'])

  # Target variable: delivery duration in minutes
  df['actual_duration_min'] = (
      df['delivery_time'] - df['dispatch_time']
  ).dt.total_seconds() / 60.0

  # Filter operational outliers
  df = df[(df['actual_duration_min'] > 1) & (df['actual_duration_min'] < 720)]

  # Temporal feature extraction
  df['dispatch_hour'] = df['dispatch_time'].dt.hour
  df['day_of_week'] = df['dispatch_time'].dt.dayofweek

  return df


def create_delivery_zones(
    df: pd.DataFrame, n_zones: int = 5
) -> tuple[pd.DataFrame, KMeans]:
  """Clusters delivery coordinates into optimized geographic dispatch zones using K-Means."""
  coords = df[['delivery_lat', 'delivery_long']].values

  kmeans = KMeans(n_clusters=n_zones, random_state=42, n_init=10)
  df['assigned_zone'] = kmeans.fit_predict(coords)

  print('=== Optimal Micro-Hub Centroids (Lat, Long) ===')
  for idx, centroid in enumerate(kmeans.cluster_centers_):
    print(f'Zone {idx}: Latitude {centroid[0]:.4f}, Longitude {centroid[1]:.4f}')

  return df, kmeans


def train_delivery_time_model(
    df: pd.DataFrame,
) -> RandomForestRegressor:
  """Trains a Random Forest Regressor to predict delivery duration."""
  features = [
      'distance_km',
      'dispatch_hour',
      'day_of_week',
      'assigned_zone',
      'parcel_weight_kg',
  ]
  X = df[features]
  y = df['actual_duration_min']

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  predictions = model.predict(X_test)
  mae = mean_absolute_error(y_test, predictions)
  r2 = r2_score(y_test, predictions)

  print('=== Model Performance Evaluation ===')
  print(f'Mean Absolute Error (MAE): {mae:.2f} minutes')
  print(f'R^2 Score: {r2:.4f}')

  return model


if __name__ == '__main__':
  print('Logistics Strategy Data Pipeline Initialized.')
  
