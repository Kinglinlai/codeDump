import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import os
import glob
import warnings

# warning is too much :(
warnings.filterwarnings('ignore')

def prepare_station_data(station_id, base_dir='NOAA_GSOD'):
    file_pattern = os.path.join(base_dir, '*', f'{station_id}.csv')
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print(f"Not found for station {station_id}")
        return None
    
    dfs = []
    for file in file_list:
        df = pd.read_csv(file, parse_dates=['DATE'])
        dfs.append(df)
    station_df = pd.concat(dfs).sort_values('DATE').reset_index(drop=True)
    
    numeric_cols = ['TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'PRCP']
    for col in numeric_cols:
        station_df[col] = (
            station_df[col].astype(str)
            .str.strip()
            .replace({'9999.9': np.nan, '999.9': np.nan, '99.99': np.nan, '': np.nan})
            .astype(float)
        )
    
    station_df['YEAR'] = station_df['DATE'].dt.year
    station_df['MONTH'] = station_df['DATE'].dt.month
    station_df['DAY'] = station_df['DATE'].dt.day
    station_df['DAY_OF_YEAR'] = station_df['DATE'].dt.dayofyear
    station_df['WEEK_OF_YEAR'] = station_df['DATE'].dt.isocalendar().week
    
    # Lag for Time Series
    station_df['TEMP_LAG1'] = station_df['TEMP'].shift(1)

    station_df = station_df.dropna(subset=['TEMP'])

    
    # Feature selection
    features = ['DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 
                'YEAR', 'MONTH', 'DAY_OF_YEAR', 'TEMP_LAG1',
                'PRCP']
    
    return station_df[features + ['TEMP']].dropna()

def train_models(station_id, base_dir='NOAA_GSOD'):

    data = prepare_station_data(station_id, base_dir)
    if data is None or len(data) < 100:
        print(f"Insufficient data for station {station_id}")
        return
    
    X = data.drop('TEMP', axis=1)
    y = data['TEMP']
    
    # Train-test (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Preprocessing pipeline
    numeric_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('scale', StandardScaler(), numeric_features)
        ],
        remainder='passthrough'
    )
    
    models = {
        'Linear Regression': LinearRegression(),
        'Polynomial Regression (deg=2)': make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            LinearRegression()
        ),
        'XGBoost': xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            early_stopping_rounds=50,
            random_state=42
        )
    }
    
    # Train and evaluate models
    results = []
    predictions = {}
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, model) in enumerate(models.items(), 1):
        # pipeline
        full_pipeline = make_pipeline(preprocessor, model)
        
        # Train model
        if name == 'XGBoost':
            preprocessor.fit(X_train)
            X_train_processed = preprocessor.transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            model.fit(
                X_train_processed, y_train,
                eval_set=[(X_test_processed, y_test)],
                verbose=False
            )
        else:
            full_pipeline.fit(X_train, y_train)
        
        # Prediction
        if name == 'XGBoost':
            y_pred = model.predict(X_test_processed)
        else:
            y_pred = full_pipeline.predict(X_test)
        
        predictions[name] = y_pred
        
        # Evaluation
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)  # MAE
        results.append({'Model': name, 'RMSE': rmse, 'R sq': r2, 'MAE': mae}) 
        
        # Plot
        plt.subplot(2, 2, i)
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(f'{name}\nRMSE: {rmse:.2f} °F, R²: {r2:.2f}')
        plt.xlabel('Actual Temperature (°F)')
        plt.ylabel('Predicted Temperature (°F)')
        plt.grid(alpha=0.3)
    
    # TS
    plt.subplot(2, 2, 4)
    sample_size = min(100, len(y_test))
    plt.plot(y_test.values[:sample_size], label='Actual', marker='o')
    
    for name, preds in predictions.items():
        plt.plot(preds[:sample_size], '--', label=name, alpha=0.8)
    
    plt.title('Actual vs Predicted Temperature (Sample)')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature (°F)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # add title
    plt.suptitle(f'Temperature Prediction Models for Station {station_id}\n'
                 f'{len(data)} samples | Test size: {len(y_test)}', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'station_{station_id}_model_results.png', dpi=150)
    
    # Print results
    results_df = pd.DataFrame(results)
    print("\nModel Evaluation Results:")
    print(results_df.to_string(index=False))
    
        

if __name__ == "__main__":
    station_id = '01033099999'  
    train_models(station_id)