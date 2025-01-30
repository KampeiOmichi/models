import os
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from joblib import Parallel, delayed
import multiprocessing
import matplotlib.dates as mdates
import ast

# ---------------------------------------------------------------------
# Global Variables and Configuration
# ---------------------------------------------------------------------
stock_symbol = "PFE"
output_folder = stock_symbol  # Folder that has the same name as the stock symbol

# Ensure the folder exists (optional safety check)
os.makedirs(output_folder, exist_ok=True)

# Read the cleaned CSV from the symbol folder
data_path = os.path.join(output_folder, f'cleaned_{stock_symbol}_data.csv')
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' column is datetime

column_name = 'adjusted_close_price'  # Column to be modeled
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[[column_name]].values)

# ---------------------------------------------------------------------
# Dataset Creation
# ---------------------------------------------------------------------
def create_dataset(data, look_back=1):
    """
    Converts a 1D time series into a dataset suitable for SVR,
    still using a 'look_back' window. Each sample is shape (look_back,).
    """
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 30
X, Y = create_dataset(scaled_data, look_back)

# Train/Test Split
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
trainX, testX = X[:train_size], X[train_size:]
trainY, testY = Y[:train_size], Y[train_size:]

# ---------------------------------------------------------------------
# Attention Layer Definition (NumPy Implementation)
# ---------------------------------------------------------------------
class AttentionLayer:
    """
    A custom 'Attention Layer' in NumPy form. Given a batch of shape
    (samples, look_back), we treat look_back as the 'time' dimension
    and apply attention weights across those timesteps.
    
    The logic mimics:
        e = tanh(X W + b)
        a = softmax(e, axis=time)
        output = sum(X * a)
    
    We'll maintain simple W, b in shape (1,1) for demonstration.
    """
    def __init__(self, look_back):
        self.look_back = look_back
        # Simple random init for demonstration
        self.W = np.random.uniform(-0.1, 0.1, (1, 1))
        self.b = np.zeros((1,))

    def _softmax(self, e):
        # e shape: (samples, look_back, 1)
        e_max = np.max(e, axis=1, keepdims=True)
        exp_e = np.exp(e - e_max)  # subtract max for numerical stability
        sum_exp = np.sum(exp_e, axis=1, keepdims=True)
        return exp_e / sum_exp

    def apply_attention(self, X_in):
        """
        X_in: shape (samples, look_back).
        Returns: shape (samples,) => single attention-weighted sum per sample
        """
        samples = X_in.shape[0]
        X_3d = X_in.reshape((samples, self.look_back, 1))

        # e = tanh(x_t * W + b) per timestep
        e = np.tanh(X_3d * self.W + self.b)  # shape: (samples, look_back, 1)

        # a = softmax(e) across time dimension
        a = self._softmax(e)  # shape: (samples, look_back, 1)

        # Weighted sum over time
        weighted_sum = np.sum(X_3d * a, axis=1)  # shape: (samples, 1)
        return weighted_sum.flatten()

# Single global attention object (optional design choice)
attention_layer = AttentionLayer(look_back=look_back)

# ---------------------------------------------------------------------
# Build & "Compile" the Model (SVR version)
# ---------------------------------------------------------------------
def build_model(gamma_value):
    """
    Creates an SVR model. We'll interpret 'gamma_value' as the gamma
    parameter for the RBF kernel.
    """
    model = SVR(kernel='rbf', C=1.0, gamma=gamma_value)
    return model

# ---------------------------------------------------------------------
# Date Ranges for MAE Calculation
# ---------------------------------------------------------------------
date_ranges = [
    ('2015-01-01', '2018-12-31'),
    ('2018-01-01', '2020-12-31'),
    ('2020-01-01', '2022-12-31'),
    ('2022-01-01', '2024-12-31')
]

# ---------------------------------------------------------------------
# MAE Calculation Function (cast to float)
# ---------------------------------------------------------------------
def calculate_mae(diff_plot, date_ranges):
    """
    Calculates MAE in diff_plot for each start-end date in date_ranges.
    Returns a list of (start_date, end_date, mae_as_float).
    """
    mae_ranges = []
    for start_date, end_date in date_ranges:
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        range_diff = diff_plot[mask]
        # Convert to plain Python float to avoid np.float64(...) in CSV
        mae_value = float(np.mean(np.abs(range_diff[~np.isnan(range_diff)])))
        mae_ranges.append((start_date, end_date, mae_value))
    return mae_ranges

# ---------------------------------------------------------------------
# Parallelized Training Function
# ---------------------------------------------------------------------
def train_model_with_lr(gamma_value):
    """
    1) Builds an SVR with gamma=gamma_value.
    2) Applies attention to trainX -> trains the model.
    3) Predicts on train/test sets (with attention).
    4) Calculates date-range-based MAE.
    Returns: (gamma_value, mae_ranges)
    """
    # 1) Build the SVR model
    model = build_model(gamma_value)

    # 2) Apply attention to training data
    att_trainX = attention_layer.apply_attention(trainX).reshape(-1, 1)
    model.fit(att_trainX, trainY)

    # 3) Predict
    att_trainPredict = model.predict(att_trainX)
    att_testX = attention_layer.apply_attention(testX).reshape(-1, 1)
    att_testPredict = model.predict(att_testX)

    # 4) Invert the scaling
    trainPredict = scaler.inverse_transform(att_trainPredict.reshape(-1, 1))
    testPredict = scaler.inverse_transform(att_testPredict.reshape(-1, 1))
    trainY_inv = scaler.inverse_transform(trainY.reshape(1, -1))
    testY_inv = scaler.inverse_transform(testY.reshape(1, -1))

    diff_train = trainPredict[:, 0] - trainY_inv[0][:len(trainPredict)]
    diff_test = testPredict[:, 0] - testY_inv[0][:len(testPredict)]

    diff_plot = np.empty_like(scaled_data)
    diff_plot[:, :] = np.nan

    trainPredictStart = look_back
    trainPredictEnd = trainPredictStart + len(diff_train)
    testPredictStart = trainPredictEnd
    testPredictEnd = testPredictStart + len(diff_test)

    diff_plot[trainPredictStart:trainPredictEnd, 0] = diff_train
    diff_plot[testPredictStart:testPredictEnd, 0] = diff_test

    mae_ranges = calculate_mae(diff_plot, date_ranges)
    return (gamma_value, mae_ranges)

# ---------------------------------------------------------------------
# Hyperparameter Search & MAE Logging (PARALLEL)
# ---------------------------------------------------------------------
if __name__ == '__main__':
    # We'll use a typical set of gamma values for RBF SVR
    gamma_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    n_jobs = multiprocessing.cpu_count()

    print(f"Starting parallel search across {len(gamma_values)} gamma values, using {n_jobs} CPUs...")

    results = Parallel(n_jobs=n_jobs)(delayed(train_model_with_lr)(g) for g in gamma_values)

    # Save results to CSV
    mae_results_path = os.path.join(output_folder, f'{stock_symbol}_mae_results.csv')
    results_df = pd.DataFrame(results, columns=['Gamma', 'MAE Ranges'])
    results_df.to_csv(mae_results_path, index=False)
    print(f"MAE results saved to {mae_results_path}")

    # ---------------------------------------------------------------------
    # Post-Processing the MAE Results
    # ---------------------------------------------------------------------
    file_path = mae_results_path
    df_csv = pd.read_csv(file_path)

    # Transform the "MAE Ranges" column into separate columns
    transformed_data = {'Gamma': df_csv['Gamma']}
    columns_set = set()

    for index, row in df_csv.iterrows():
        mae_list = ast.literal_eval(row['MAE Ranges'])
        for date_range in mae_list:
            start_date, end_date, mae_value = date_range
            col_name = f'{start_date} to {end_date}'
            columns_set.add(col_name)
            if col_name not in transformed_data:
                transformed_data[col_name] = [None] * len(df_csv)
            transformed_data[col_name][index] = mae_value

    for col in columns_set:
        if col not in transformed_data:
            transformed_data[col] = [None] * len(df_csv)

    transformed_df = pd.DataFrame(transformed_data)
    transformed_df.to_csv(file_path, index=False)

    # If you really have 2 extra lines in your CSV, keep skipfooter=2; otherwise remove or set skipfooter=0
    df_csv = pd.read_csv(file_path, skipfooter=2, engine='python')
    mean_values = df_csv.iloc[:, 1:].mean(axis=1)
    median_values = df_csv.iloc[:, 1:].median(axis=1)

    df_csv['Mean'] = mean_values
    df_csv['Median'] = median_values
    df_csv.to_csv(file_path, index=False)

    df_csv = pd.read_csv(file_path)
    df_csv['Mean'] = df_csv.iloc[:, 1:-2].mean(axis=1)
    df_csv['Median'] = df_csv.iloc[:, 1:-2].median(axis=1)

    min_mean_row = df_csv.loc[df_csv['Mean'].idxmin()]
    min_median_row = df_csv.loc[df_csv['Median'].idxmin()]

    min_mean_gamma = min_mean_row['Gamma']
    min_mean_value = min_mean_row['Mean']
    min_median_gamma = min_median_row['Gamma']
    min_median_value = min_median_row['Median']

    print(f"\nBest Gamma by Mean: {min_mean_gamma}, Value: {min_mean_value}")
    print(f"Best Gamma by Median: {min_median_gamma}, Value: {min_median_value}")

    # ---------------------------------------------------------------------
    # Retrain with Best Gamma and Visualize (SVR + Attention)
    # ---------------------------------------------------------------------
    df_final = pd.read_csv(data_path)
    Gamma = min_median_gamma

    df_final['Date'] = pd.to_datetime(df_final['Date'])
    column_name = 'adjusted_close_price'
    scaler_final = MinMaxScaler(feature_range=(0, 1))
    scaled_data_final = scaler_final.fit_transform(df_final[[column_name]].values)

    def create_dataset_final(data, look_back=1):
        X_out, Y_out = [], []
        for i in range(len(data) - look_back):
            X_out.append(data[i:(i + look_back), 0])
            Y_out.append(data[i + look_back, 0])
        return np.array(X_out), np.array(Y_out)

    X_final, Y_final = create_dataset_final(scaled_data_final, look_back)
    train_size = int(len(X_final) * 0.8)
    test_size = len(X_final) - train_size
    trainX_final, testX_final = X_final[:train_size], X_final[train_size:]
    trainY_final, testY_final = Y_final[:train_size], Y_final[train_size:]

    att_trainX_final = attention_layer.apply_attention(trainX_final).reshape(-1, 1)

    model_final = build_model(Gamma)
    model_final.fit(att_trainX_final, trainY_final)

    trainPredict_final = model_final.predict(att_trainX_final)
    att_testX_final = attention_layer.apply_attention(testX_final).reshape(-1, 1)
    testPredict_final = model_final.predict(att_testX_final)

    trainPredict_final = scaler_final.inverse_transform(trainPredict_final.reshape(-1, 1))
    testPredict_final = scaler_final.inverse_transform(testPredict_final.reshape(-1, 1))
    trainY_inv_final = scaler_final.inverse_transform(trainY_final.reshape(1, -1))
    testY_inv_final = scaler_final.inverse_transform(testY_final.reshape(1, -1))

    trainPredictStart = look_back
    trainPredictEnd = trainPredictStart + len(trainPredict_final)
    testPredictStart = trainPredictEnd
    testPredictEnd = testPredictStart + len(testPredict_final)

    trainPredictPlot = np.empty_like(scaled_data_final)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[trainPredictStart:trainPredictEnd, 0] = trainPredict_final[:, 0]

    testPredictPlot = np.empty_like(scaled_data_final)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[testPredictStart:testPredictEnd, 0] = testPredict_final[:, 0]

    diff_train_final = trainPredict_final[:, 0] - trainY_inv_final[0][:len(trainPredict_final)]
    diff_test_final = testPredict_final[:, 0] - testY_inv_final[0][:len(testPredict_final)]
    diff_plot_final = np.empty_like(scaled_data_final)
    diff_plot_final[:, :] = np.nan

    diff_plot_final[trainPredictStart:trainPredictEnd, 0] = diff_train_final
    diff_plot_final[testPredictStart:testPredictEnd, 0] = diff_test_final

    average_error_final = np.mean(np.abs(diff_plot_final[~np.isnan(diff_plot_final)]))
    print(f"\nFinal model average error (MAE): {average_error_final:.4f}")

    # Plot actual vs. predicted
    fig_final, (ax1_final, ax2_final) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    ax1_final.plot(
        df_final['Date'],
        scaler_final.inverse_transform(scaled_data_final),
        label='Actual Adj Close',
        color='b'
    )
    ax1_final.plot(
        df_final['Date'], trainPredictPlot,
        label='Train Predict',
        color='r'
    )
    ax1_final.plot(
        df_final['Date'], testPredictPlot,
        label='Test Predict',
        color='g'
    )
    ax1_final.set_title('Actual vs Predicted Stock Prices (SVR + Attention)')
    ax1_final.set_ylabel('Adjusted Close Price')
    ax1_final.legend()

    ax2_final.plot(
        df_final['Date'], diff_plot_final,
        label='Difference (Predicted - Actual)',
        color='m'
    )
    ax2_final.axhline(y=0, color='k')
    ax2_final.set_title('Difference Between Actual and Predicted Stock Prices')
    ax2_final.set_xlabel('Date')
    ax2_final.set_ylabel('Difference')
    ax2_final.legend()

    ax2_final.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax2_final.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("\nExecution complete with parallelized hyperparameter search, using SVR + Attention.")
