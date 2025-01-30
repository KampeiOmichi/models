import os
import re
import ast
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Keras / TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

from joblib import Parallel, delayed
import multiprocessing
import matplotlib.dates as mdates

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
    Converts a 1D time series into a dataset suitable for a CNN,
    shaping each sample as (look_back, 1).
    """
    X, Y = [], []
    for i in range(len(data) - look_back):
        # The feature vector is the previous 'look_back' values
        X.append(data[i:(i + look_back), 0])
        # The label is the value right after that window
        Y.append(data[i + look_back, 0])
    # Reshape X to (samples, look_back, 1) for Conv1D
    X = np.array(X).reshape(-1, look_back, 1)
    Y = np.array(Y)
    return X, Y

look_back = 30
X, Y = create_dataset(scaled_data, look_back)

# Train/Test Split
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
trainX, testX = X[:train_size], X[train_size:]
trainY, testY = Y[:train_size], Y[train_size:]

# ---------------------------------------------------------------------
# CNN Model (Build & Compile)
# ---------------------------------------------------------------------
def build_model(learning_rate):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(look_back, 1)),
        Flatten(),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )
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
# MAE Calculation Function
# ---------------------------------------------------------------------
def calculate_mae(diff_plot, date_ranges):
    """
    Calculates MAE in diff_plot for each start-end date in date_ranges,
    returning tuples of (start_date, end_date, mae) with plain floats.
    """
    mae_ranges = []
    for start_date, end_date in date_ranges:
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        range_diff = diff_plot[mask]

        # Convert to float to avoid np.float64(...) in your CSV
        mae_value = float(np.mean(np.abs(range_diff[~np.isnan(range_diff)])))
        mae_ranges.append((start_date, end_date, mae_value))

    return mae_ranges

# ---------------------------------------------------------------------
# Parallelized Training Function
# ---------------------------------------------------------------------
def train_model_with_lr(lr):
    """
    1) Builds a CNN with the given learning rate.
    2) Trains on trainX, trainY for a fixed number of epochs/batch_size.
    3) Predicts on train/test sets.
    4) Calculates date-range-based MAE.
    Returns: (lr, mae_ranges) as Python-literal-friendly data.
    """
    # Build the CNN model
    model = build_model(lr)

    # Train quietly for a fixed epoch count, e.g., 50
    model.fit(
        trainX, trainY,
        epochs=50,
        batch_size=64,
        verbose=0
    )

    # Generate predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Invert predictions and labels
    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)
    trainY_inv = scaler.inverse_transform(trainY.reshape(-1, 1))
    testY_inv = scaler.inverse_transform(testY.reshape(-1, 1))

    # Prepare the difference array for MAE
    diff_train = trainPredict[:, 0] - trainY_inv[:, 0]
    diff_test = testPredict[:, 0] - testY_inv[:, 0]

    diff_plot = np.empty_like(scaled_data)
    diff_plot[:, :] = np.nan

    # Align predictions with the actual data timeline
    trainPredictStart = look_back
    trainPredictEnd = trainPredictStart + len(diff_train)
    testPredictStart = trainPredictEnd
    testPredictEnd = testPredictStart + len(diff_test)

    diff_plot[trainPredictStart:trainPredictEnd, 0] = diff_train
    diff_plot[testPredictStart:testPredictEnd, 0] = diff_test

    mae_ranges = calculate_mae(diff_plot, date_ranges)

    # Return the learning rate and a Python-literal-friendly list of tuples
    return (lr, mae_ranges)

# ---------------------------------------------------------------------
# Hyperparameter Search & MAE Logging (PARALLEL)
# ---------------------------------------------------------------------
if __name__ == '__main__':
    learning_rates = [
        0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 
        0.0006, 0.0007, 0.0008, 0.0009, 0.001, 
        0.002, 0.005, 0.01, 0.02, 0.05, 
        0.1, 0.15, 0.2, 0.25, 0.5
    ]
    n_jobs = multiprocessing.cpu_count()
    print(f"Starting parallel search across {len(learning_rates)} CNN learning rates, using {n_jobs} CPUs...")

    results = Parallel(n_jobs=n_jobs)(delayed(train_model_with_lr)(lr) for lr in learning_rates)

    # Save results to CSV inside the stock symbol folder
    mae_results_path = os.path.join(output_folder, f'{stock_symbol}_mae_results.csv')
    results_df = pd.DataFrame(results, columns=['Learning Rate', 'MAE Ranges'])
    results_df.to_csv(mae_results_path, index=False)
    print(f"MAE results saved to {mae_results_path}")

# ---------------------------------------------------------------------
# Post-Processing the MAE Results
# ---------------------------------------------------------------------

# Read the results CSV again
df_csv = pd.read_csv(mae_results_path)

# We'll transform the "MAE Ranges" column into multiple columns
transformed_data = {'Learning Rate': df_csv['Learning Rate']}
columns_set = set()

for index, row in df_csv.iterrows():
    mae_str = row['MAE Ranges']
    # Parse as Python literal: e.g. "[('2015-01-01', '2018-12-31', 1.23), ...]"
    try:
        mae_list = ast.literal_eval(mae_str)  # <--- Use literal_eval instead of json
        for (start_date, end_date, mae_value) in mae_list:
            col_name = f'{start_date} to {end_date}'
            columns_set.add(col_name)
            if col_name not in transformed_data:
                transformed_data[col_name] = [None] * len(df_csv)
            transformed_data[col_name][index] = mae_value
    except (SyntaxError, ValueError) as e:
        print(f"Error processing row {index}: {e}")
        print(f"Problematic data: {mae_str}")
        continue

# Convert transformed_data to a DataFrame and overwrite the CSV
transformed_df = pd.DataFrame(transformed_data)
transformed_df.to_csv(mae_results_path, index=False)

# Now we have columns for each date range. We'll compute Mean/Median of those columns.
df_csv = pd.read_csv(mae_results_path)

# The first column is "Learning Rate", so the subsequent columns are the date ranges
date_range_cols = df_csv.columns[1:]  # from 1 onward

# Compute row-wise Mean & Median
df_csv['Mean'] = df_csv[date_range_cols].mean(axis=1)
df_csv['Median'] = df_csv[date_range_cols].median(axis=1)

# Save it back
df_csv.to_csv(mae_results_path, index=False)

# Identify the row with the lowest Mean
if df_csv['Mean'].isna().all():
    print("All Mean values are NaN—cannot compute best LR by Mean.")
    min_mean_learning_rate = None
    min_mean_value = None
else:
    min_mean_row = df_csv.loc[df_csv['Mean'].idxmin()]
    min_mean_learning_rate = min_mean_row['Learning Rate']
    min_mean_value = min_mean_row['Mean']

# Identify the row with the lowest Median
if df_csv['Median'].isna().all():
    print("All Median values are NaN—cannot compute best LR by Median.")
    min_median_learning_rate = None
    min_median_value = None
else:
    min_median_row = df_csv.loc[df_csv['Median'].idxmin()]
    min_median_learning_rate = min_median_row['Learning Rate']
    min_median_value = min_median_row['Median']

print(f"\nBest LR by Mean: {min_mean_learning_rate}, Value: {min_mean_value}")
print(f"Best LR by Median: {min_median_learning_rate}, Value: {min_median_value}")

# ---------------------------------------------------------------------
# Retrain with Best LR and Visualize (CNN)
# ---------------------------------------------------------------------

# For safety, pick whichever metric you prefer:
LR = min_median_learning_rate if min_median_learning_rate is not None else 0.001

df_final = pd.read_csv(data_path)
df_final['Date'] = pd.to_datetime(df_final['Date'])

scaler_final = MinMaxScaler(feature_range=(0, 1))
scaled_data_final = scaler_final.fit_transform(df_final[[column_name]].values)

def create_dataset_final(data, look_back=1):
    X_out, Y_out = [], []
    for i in range(len(data) - look_back):
        X_out.append(data[i:(i + look_back), 0])
        Y_out.append(data[i + look_back, 0])
    X_out = np.array(X_out).reshape(-1, look_back, 1)
    Y_out = np.array(Y_out)
    return X_out, Y_out

X_final, Y_final = create_dataset_final(scaled_data_final, look_back)
train_size = int(len(X_final) * 0.8)
test_size = len(X_final) - train_size
trainX_final, testX_final = X_final[:train_size], X_final[train_size:]
trainY_final, testY_final = Y_final[:train_size], Y_final[train_size:]

def build_final_model(learning_rate):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(look_back, 1)))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

model_final = build_final_model(LR)
model_final.fit(
    trainX_final, trainY_final,
    epochs=50,
    batch_size=64,
    verbose=2
)

trainPredict_final = model_final.predict(trainX_final)
testPredict_final = model_final.predict(testX_final)

trainPredict_final = scaler_final.inverse_transform(trainPredict_final)
testPredict_final = scaler_final.inverse_transform(testPredict_final)
trainY_inv_final = scaler_final.inverse_transform(trainY_final.reshape(-1, 1))
testY_inv_final = scaler_final.inverse_transform(testY_final.reshape(-1, 1))

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

diff_train_final = trainPredict_final[:, 0] - trainY_inv_final[:, 0]
diff_test_final = testPredict_final[:, 0] - testY_inv_final[:, 0]
diff_plot_final = np.empty_like(scaled_data_final)
diff_plot_final[:, :] = np.nan

diff_plot_final[trainPredictStart:trainPredictEnd, 0] = diff_train_final
diff_plot_final[testPredictStart:testPredictEnd, 0] = diff_test_final

average_error_final = np.mean(np.abs(diff_plot_final[~np.isnan(diff_plot_final)]))
print(f"\nFinal model average error (MAE): {average_error_final:.4f}")

# Plot actual vs predicted
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
ax1_final.set_title('Actual vs Predicted Stock Prices (CNN)')
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

print("\nExecution complete with parallelized hyperparameter search, using a CNN.")
