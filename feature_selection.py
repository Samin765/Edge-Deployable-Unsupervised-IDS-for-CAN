import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler


def binary_encode(ids, num_bits):
    """
    Converts a list of CAN IDs (hexadecimal strings) to binary representation.
    
    Args:
        ids (list of str): List of CAN IDs as hexadecimal strings (e.g., ['0x101', '0x102']).
        num_bits (int): Number of bits to represent the IDs in binary format.
        
    Returns:
        numpy.ndarray: Array of binary representations, where each row is a binary vector.
    """
    binary_ids = []
    for id_str in ids:
        # Convert hexadecimal string to integer
        id_int = int(id_str, 16)
        # Convert integer to binary and pad with leading zeros
        binary_vector = [int(bit) for bit in f"{id_int:0{num_bits}b}"]
        binary_ids.append(binary_vector)
    return np.array(binary_ids)


def binary_encode_integers(ids, num_bits):
    """
    Converts a list of CAN IDs (integers) to binary representation.

    Args:
        ids (list of int): List of CAN IDs as integers (e.g., [452948266, 452946218]).
        num_bits (int): Number of bits to represent the IDs in binary format.

    Returns:
        numpy.ndarray: Array of binary representations, where each row is a binary vector.
    """
    binary_ids = []
    for id_int in ids:
        # Convert integer to binary and pad with leading zeros
        binary_vector = [int(bit) for bit in f"{id_int:0{num_bits}b}"]
        binary_ids.append(binary_vector)
    return np.array(binary_ids)


def calculate_entropy(row):
    byte_values = [row[f'data[{i}]'] for i in range(8)]
    # Compute frequency of each byte value (0-255)
    counts = np.bincount(byte_values, minlength=256)
    probabilities = counts / np.sum(counts)
    return entropy(probabilities, base=2)


def hamming_distance(payload1, payload2):
    """Calculate Hamming distance between two 8-byte payloads."""
    # Convert payloads to binary strings
    bin1 = ''.join(f"{int(byte, 16):08b}" for byte in payload1)
    bin2 = ''.join(f"{int(byte, 16):08b}" for byte in payload2)

    # Count differing bits
    return sum(b1 != b2 for b1, b2 in zip(bin1, bin2))


def compute_hamming_distances(dataframe, scaler, previous_x=1):
    """
    Computes the Hamming distance between each row's payload and the previous X messages.
    
    Args:
        dataframe (pd.DataFrame): CAN dataset with 'data' column.
        previous_x (int): Number of previous messages to compare with.

    Returns:
        pd.Series: A series containing the Hamming distances.
    """
    hamming_distances = []
    data_columns = [f'data[{i}]' for i in range(8)]  # Extracted byte columns

    for i in range(len(dataframe)):
        if i < previous_x:
            hamming_distances.append(0)  # No previous message to compare with
        else:
            prev_payload = dataframe.iloc[i - previous_x][data_columns].astype(str).tolist()
            curr_payload = dataframe.iloc[i][data_columns].astype(str).tolist()
            hamming_distances.append(hamming_distance(prev_payload, curr_payload))

    dataframe['hamming_distance'] = hamming_distances
    return dataframe


def compute_temporal_features_tsfresh(dataframe, custom_fc_parameters = None, ts_fresh = False):
    """
    Computes tsfresh features while preserving message-level granularity.
    
    Args:
        dataframe (pd.DataFrame): CAN data with 'timestamp', 'arbitration_id', 'data'.
        window_size (int): Rolling window size for local statistics.
    
    Returns:
        pd.DataFrame: Original dataframe with merged tsfresh features.
    """
    #dataframe = dataframe.copy()
    dataframe.sort_values(by="timestamp")
    # Check if timestamp is already a float and handle accordingly
    if dataframe["timestamp"].dtype != 'float64':
        # Convert the 'timestamp' column to datetime format and handle errors
        print("dates not floats")
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], errors='coerce')

    # Check for NaT values and print those rows for inspection
    if dataframe["timestamp"].isna().sum() > 0:
        print(f"Found {dataframe['timestamp'].isna().sum()} NaT values in timestamp. Rows: {dataframe[dataframe['timestamp'].isna()]}")

    # Compute Inter-Arrival Time (IAT)
    dataframe["iat"] = dataframe.groupby("arbitration_id")["timestamp"].diff().fillna(0)

    # Check for NaN values in 'iat' and handle
    if dataframe["iat"].isna().sum() > 0:
        print("NaN values found in 'iat' column")
        dataframe["iat"] = dataframe["iat"].fillna(0)  # Replace NaNs with 0 or other logic

    # Compute Rolling Window Statistics per message
    window_size_seconds = 0.2 # interval between log 1 and 50 is approx 0.18seconds
    timestamps = dataframe["timestamp"].to_numpy()
    arbitration_ids = dataframe["arbitration_id"].to_numpy()

    # Initialize the result array
    msg_count_last_20ms = np.zeros(len(dataframe), dtype=int)
    payload_entropy_last_20ms = np.zeros(len(dataframe), dtype=int)

    # Define the window size in seconds (50 milliseconds = 0.050 seconds)

    # Iterate through each row and compute the count
    for i, (current_time, current_id) in enumerate(zip(timestamps, arbitration_ids)):
        mask = (timestamps >= current_time - window_size_seconds) & \
            (timestamps <= current_time) & \
            (arbitration_ids == current_id)
        msg_count_last_20ms[i] = np.sum(mask)

    # Add the result to the dataframe
    dataframe["msg_frequency"] = msg_count_last_20ms


    # Normalize msg_count_last_50ms to [0, 1]
    max_count = dataframe["msg_frequency"].max()
    if max_count > 0:
        dataframe["msg_frequency"] = dataframe["msg_frequency"] / max_count
    else:
        dataframe["msg_frequency"] = 0  # If max_count is 0, set all normalized values to 0

    dataframe["rolling_mean_iat"] = dataframe.groupby("arbitration_id")["iat"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    dataframe["rolling_std_iat"] = dataframe.groupby("arbitration_id")["iat"].transform(lambda x: x.rolling(20, min_periods=1).std().fillna(0))

    # Check for NaN values in 'id' or 'time' columns and handle
    if dataframe['arbitration_id'].isna().sum() > 0 or dataframe['timestamp'].isna().sum() > 0:
        print("NaN values found in 'id' or 'time' columns")
        dataframe = dataframe.dropna(subset=['arbitration_id', 'timestamp'])

    if ts_fresh:
        # Prepare for tsfresh
        tsfresh_df = dataframe.rename(columns={"arbitration_id": "id", "timestamp": "time", "iat": "value"})
        tsfresh_df = tsfresh_df[['id', 'time', 'value']]


        if custom_fc_parameters is None:
            custom_fc_parameters = EfficientFCParameters()

        
        extracted_features = extract_features(tsfresh_df, column_id="id", column_sort="time", 
                                        default_fc_parameters=custom_fc_parameters)
        
        # Impute missing values
        impute(extracted_features)

        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(extracted_features)
        normalized_extracted_features = pd.DataFrame(normalized_features, columns=extracted_features.columns, index=extracted_features.index)
        
        # Merge extracted features back to the original dataframe without losing time-series order
        dataframe = dataframe.merge(normalized_extracted_features, left_on="arbitration_id", right_index=True, how="left")
        dataframe.sort_values(by="timestamp")
    return dataframe


def feature_selection_preparation(file_name, phase ,pre_dataframe = None, rows = None, ts_fresh = False, ts_fresh_parameters = None, ts_fresh_custom_features = None):
    column_names_train = ['timestamp' , 'arbitration_id' , 'channel' , 'dlc', 'data' , 'ecu']
    column_names_test = ['timestamp', 'arbitration_id', 'dlc', 'data']
    # Define consistent dtypes for all phases
    # First check original file data types (read a few rows)
    print(f"Checking original data types in {file_name}...")
    try:
        sample_df = pd.read_csv(file_name, nrows=1)
        print("Original inferred data types:")
        for col in sample_df.columns:
            print(f"  {col}: {sample_df[col].dtype}")

        # To see the actual value format
        print("\nSample values for each column:")
        for col in sample_df.columns:
            print(f"  {col}: {sample_df[col].iloc[0]} (type: {type(sample_df[col].iloc[0]).__name__})")
    except Exception as e:
        print(f"Error sampling file: {e}")

    dtypes_train = {
        'timestamp': float,
        'arbitration_id': int,
        'channel': int,
        'dlc': int,
        'data': str,
        'ecu': str

    }

    dtypes_test = {
        'timestamp': float,
        'arbitration_id': int,
        'dlc': int,
        'data': str,
        'type': str
    }

    try:
        if phase == 'training':
            dataframe = pd.read_csv(
                file_name, 
                header=0, 
                names=column_names_train, 
                nrows=rows, 
                dtype=dtypes_train
            )
            
        elif phase == 'debug':
            dataframe = pre_dataframe
            
        elif phase == 'test':
            # Read the CSV with consistent dtypes
            dataframe = pd.read_csv(
                file_name, 
                header=0, 
                names=column_names_test + ['type'], 
                nrows=rows, 
                dtype=dtypes_test
            )
            
            # Explicitly convert arbitration_id to integers
            # First handle any potential NaN values
            dataframe['arbitration_id'] = dataframe['arbitration_id'].fillna(0)
            
            # Then convert to integers
            dataframe['arbitration_id'] = dataframe['arbitration_id'].astype(int)

            # Create a copy before modifying to avoid SettingWithCopyWarning
            dataframe = dataframe.copy()
            
            # Count types before processing for debugging
            print(f"Raw type values: {dataframe['type'].unique()}")
            
            # Replace missing values with a consistent value before conversion
            dataframe['type'] = dataframe['type'].fillna('unknown')
            
            # Explicitly convert only 'R' to 0 and everything else to 1
            dataframe['type'] = dataframe['type'].apply(lambda x: 0 if x == 'R' else 1)
            
            # Print the count of anomalies
            count_of_ones = dataframe['type'].sum()
            print(f"Anomalies in 'type' column: {count_of_ones}")
            
        else:
            print("Invalid phase")
            return None
            
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None
    
    scaler = MinMaxScaler()

    #print(dataframe.head(2))
    dataframe = compute_temporal_features_tsfresh(dataframe, custom_fc_parameters= ts_fresh_custom_features, ts_fresh= ts_fresh)
    #dataframe = compute_temporal_features(dataframe)
    
    dataframe = dataframe[dataframe['dlc'] == 8].reset_index(drop=True)

    # Extract data to indiviual columns
    data_columns = [f'data[{i}]' for i in range(8)]
    dataframe[data_columns] = dataframe['data'].str.split(' ', expand=True).iloc[:, :8]

    dataframe = compute_hamming_distances(dataframe, scaler,previous_x=5)  # Compare with the last 5 messages
    
    # Convert Data from Hexadecimal to Integers [0,255]
    for col in data_columns:
        dataframe[col] = dataframe[col].apply(lambda x: int(x, 16) if isinstance(x, str) else x)

    # Entropy feature
    dataframe['payload_entropy'] = dataframe.apply(calculate_entropy, axis=1)    

    num_bits = 29  # Standard for CAN IDs
    binary_encoded_ids = binary_encode_integers(dataframe['arbitration_id'].tolist(), num_bits)
    binary_encoded_df = pd.DataFrame(binary_encoded_ids, columns=[f'bit_{i}' for i in range(num_bits)])

    # Add binary-encoded IDs to the original DataFrame
    #dataframe = dataframe.drop(columns=['arbitration_id']).reset_index(drop=True)
    dataframe = pd.concat([binary_encoded_df, dataframe], axis=1)
    
    dataframe[data_columns] = scaler.fit_transform(dataframe[data_columns])
    dataframe['timestamp'] = scaler.fit_transform(dataframe[['timestamp']])
    dataframe['payload_entropy'] = scaler.fit_transform(dataframe[['payload_entropy']])
    dataframe['hamming_distance'] = scaler.fit_transform(dataframe[['hamming_distance']])


    # Create a combined feature column, ensuring everything is a float
    #dataframe['features'] = dataframe.apply(lambda row: np.concatenate([row[[f'bit_{i}' for i in range(num_bits)]].values , row[data_columns].values]), axis=1)
    if ts_fresh:
        dataframe['features'] = dataframe.apply(
            lambda row: np.concatenate([
                row[[f'bit_{i}' for i in range(num_bits)]].values, 
                row[data_columns].values, 
                row[ts_fresh_parameters].values
            ]), axis=1
        )
    else:
        dataframe['features'] = dataframe.apply(
        lambda row: np.concatenate(
            [row[[f'bit_{i}' for i in range(num_bits)]].values,  # Binary data columns 0 to 28
            row[data_columns].values,  # Other data columns 28 to 36
            np.array([row['msg_frequency'], row['iat'],row['payload_entropy'],row['hamming_distance'],row['timestamp']])  # Additional features
            ]), axis=1)

        
        # row['iat'], row['msg_frequency'],  row['rolling_mean_iat'] , row['rolling_std_iat'],
        #   np.array([ row['payload_entropy'],   
        #              row['hamming_distance']
        #              ])]), axis=1)
        #row['payload_entropy'], row['hamming_distance'], 
        
    nan_counts = dataframe.isna().sum()
    if nan_counts.any():
        print("NaN values found:\n", nan_counts[nan_counts > 0])
    print(np.any(np.isinf(dataframe.select_dtypes(include=[np.number]))))
    return dataframe


def create_sliding_windows(data, labels=None, window_size=5, stride=1, anomaly_window_ratio = 0.5):
    # Generates sliding windows for both features and labels.
    X = np.array([data[i:i+window_size] for i in range(0, len(data) - window_size + 1, stride)], dtype=np.float32)
    print("Original window that works: ", X.shape)  # (991, 50, 40)
    """
    # Create array with the same structure but with added features
    num_window_features = 3
    augmented_X = np.zeros((X.shape[0], X.shape[1], X.shape[2] + num_window_features), dtype=np.float32)

    # Copy all original data (already normalized, so we keep as is)
    augmented_X[:, :, :X.shape[2]] = X

    # Calculate window features and add to each data point in that window
    for i, window in enumerate(X):
        # Split window into meaningful components
        arbitration_ids = window[:, :29]  # First 29 bits (arbitration ID)
        payloads = window[:, 29:37]       # Next 8 bits (payload)
        entropy_values = window[:, 37]    # Next value (entropy)
        hamming_distances = window[:, 38] # Next value (hamming distance)
        timestamps = window[:, 39]        # Last value (timestamp)
    
        # Calculate window-level features
        unique_arbitration_ids = len(np.unique(arbitration_ids, axis=0))
        mean_entropy = np.mean(entropy_values)
        var_entropy = np.var(entropy_values)
        mean_hamming = np.mean(hamming_distances)
        inter_arrival_times = np.diff(timestamps)
        mean_inter_arrival = np.mean(inter_arrival_times) if len(inter_arrival_times) > 0 else 0
        
        # Store window features temporarily
        window_features = np.array([
            #unique_arbitration_ids,
            mean_entropy,
            #var_entropy,
            mean_hamming,
            mean_inter_arrival
        ])

        #print("mean entropy", mean_entropy)
        #print("mean IAT", mean_inter_arrival)

        # Add these window features to all data points in this window
        for j in range(num_window_features):
            augmented_X[i, :, X.shape[2] + j] = window_features[j]

    print("Shape after adding window features:", augmented_X.shape)
    X = augmented_X
    """
    
    if labels is not None:
        labels = labels.values

        # Initialize an empty list to store the labels
        y = []

        # Define threshold: At least 50% of the window should contain 1s
        threshold = anomaly_window_ratio * window_size

        for i in range(0, len(labels) - window_size + 1, stride):
            # Extract the current window of labels
            window = labels[i:i+window_size]
            # Check if there is at least one '1' in the window
            if np.sum(window == 1) >= threshold:  # Count 1s and compare to threshold
                y.append(1)
            else:
                y.append(0)  # If there's not enough 1s in window, mark this window as normal

        # Convert the list to a numpy array
        y = np.array(y, dtype=np.float32)
        count_of_ones = np.sum(y)
        print(f"Anomalies in 'y' array: {count_of_ones}")
        return X, y
    return X



def convert_to_tensorflow(featureframe, labels=None, batch_size=32, window_size=5, stride=1, split_ratio = 0.8, window_anomaly_ratio = 0.5):
    # Convert feature list to NumPy array
    input_data = np.array(featureframe.tolist(), dtype=np.float32)
    before_window_shape = input_data.shape

    # Check if Train or Test dataframe
    if labels is not None:
        input_data, labels = create_sliding_windows(input_data, labels, window_size, stride, window_anomaly_ratio)
        print("x before tensor", len(input_data))
        print("y before tensor", len(labels))
        # todo: check windows 
        labels = np.array(labels, dtype=np.float32)  # Ensure labels are NumPy arrays
        model_input = tf.data.Dataset.from_tensor_slices((input_data, labels))

        dataset_size = len(input_data)
        test_size = int(dataset_size*split_ratio)

        #Shuffle , decrease size of buffer for faster shuffling
        #buffer_size=min(50000, dataset_size)
        #model_input = model_input.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=False, seed= SEED)

        # Split to train and anomaly threshold set
        train_dataset = model_input.take(test_size)
        val_dataset = model_input.skip(test_size)

        # Batch
        test_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        print(f"Feature shape BEFORE sliding window: {before_window_shape}")
        print(f"Feature shape AFTER sliding window: {input_data.shape}")
        return test_dataset, val_dataset  

    else:
        input_data = create_sliding_windows(input_data, labels=None, window_size=window_size, stride=stride)
        model_input = tf.data.Dataset.from_tensor_slices(input_data)

        # Get Size of training set
        dataset_size = len(input_data)
        train_size = int(dataset_size * split_ratio)
        
        #Shuffle , decrease size of buffer for faster shuffling
        #buffer_size=min(1000, dataset_size)
        #model_input = model_input.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True, seed= SEED)

        # Split to train and anomaly threshold set
        train_dataset = model_input.take(train_size)
        val_dataset = model_input.skip(train_size)

        # Batch
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        print(f"Feature shape BEFORE sliding window: {before_window_shape}")
        print(f"Feature shape AFTER sliding window: {input_data.shape}")
        return train_dataset, val_dataset  

    # Apply batching
    model_input = model_input.batch(batch_size)
    print(f"Feature shape BEFORE sliding window: {before_window_shape}")
    print(f"Feature shape AFTER sliding window: {input_data.shape}")
    print(f"Successfully prepared model input data.")
    return model_input
