import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from collections import defaultdict
import time
from id_embedding import train_embedding


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


def compute_temporal_features(dataframe, custom_fc_parameters = None, ts_fresh = False):
    """
    Computes tsfresh features while preserving message-level granularity.
    
    Args:
        dataframe (pd.DataFrame): CAN data with 'timestamp', 'arbitration_id', 'data'.
        window_size (int): Rolling window size for local statistics.
    
    Returns:
        pd.DataFrame: Original dataframe with merged tsfresh features.
    """
    start_time = time.time()
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

    # Compute Rolling Window Statistics per message
    window_size_seconds = 0.2 # interval between log 1 and 50 is approx 0.18seconds
    timestamps = dataframe["timestamp"].to_numpy()
    arbitration_ids = dataframe["arbitration_id"].to_numpy()
    # Get pre-calculated entropy values
    payload_entropy = dataframe['payload_entropy'].values

    # Initialize the result array
    msg_count_last_20ms = np.zeros(len(dataframe), dtype=int)
    payload_entropy_last_20ms = np.zeros(len(dataframe), dtype=int)
    payload_change_rate = np.zeros(len(timestamps))
    id_specific_payload_entropy = np.zeros(len(timestamps))
    id_specific_payload_entropy_std = np.zeros(len(timestamps))
    id_specific_payload_entropy_std = np.zeros(len(timestamps))
    id_specific_payload_entropy_z_score = np.zeros(len(timestamps))


    # Pre-extract all payload data into numpy arrays for faster access (only needed for change rate)
    payload_bytes_matrix = np.zeros((len(dataframe), 8), dtype=np.uint8)
    for j in range(8):
        payload_bytes_matrix[:, j] = dataframe[f'data[{j}]'].values

    # Define the window size in seconds (50 milliseconds = 0.050 seconds)
    # Iterate through each row and compute the count
    # Initialize timing variables
    id_timing = []
    payload_average_timing = []
    payload_change_timing = []
    payload_id_timing = []

    for i, (current_time, current_id) in enumerate(zip(timestamps, arbitration_ids)):

        start_time_id = time.time()
        # ID Statistics
        mask = (timestamps >= current_time - window_size_seconds) & \
            (timestamps <= current_time) & \
            (arbitration_ids == current_id)
        msg_count_last_20ms[i] = np.sum(mask)

        end_time_id = time.time()
        id_timing.append(end_time_id - start_time_id)

        start_time_payload_id = time.time()
        id_specific_window_indices = np.where(mask)[0]

        if len(id_specific_window_indices) > 1:
            # Get average entropy for this specific ID in the time window
            id_specific_payload_entropy[i] = np.mean(payload_entropy[id_specific_window_indices])
            
            # Get standard deviation of entropy for this specific ID (useful for detecting anomalies)
            id_specific_payload_entropy_std[i] = np.std(payload_entropy[id_specific_window_indices])
            
            # Calculate how current message entropy deviates from historical values for this ID
            current_entropy = payload_entropy[i]
            historical_mean = id_specific_payload_entropy[i]
            historical_std = id_specific_payload_entropy_std[i]

            if historical_std > 0:  # Avoid division by zero
                id_specific_payload_entropy_z_score[i] = (current_entropy - historical_mean) / historical_std

        end_time_payload_id = time.time()
        payload_id_timing.append(end_time_payload_id - start_time_payload_id)

        start_time_average_payload = time.time()
        # Payload Statistics
        time_mask = (timestamps >= current_time - window_size_seconds) & \
                    (timestamps <= current_time)
        window_indices = np.where(time_mask)[0]
    
        end_time_average_payload = time.time()
        payload_average_timing.append(end_time_average_payload - start_time_average_payload)

        # Skip entropy calculation if only one message in window

        start_time_change_payload = time.time()
 
        if len(window_indices) > 1:
            # Get average of pre-calculated entropy values
            payload_entropy_last_20ms[i] = np.mean(payload_entropy[window_indices])
            
            # Calculate payload change rate
            # Sort messages by timestamp
            sorted_indices = np.argsort(timestamps[window_indices])
            sorted_window_indices = window_indices[sorted_indices]
            sorted_payloads = payload_bytes_matrix[sorted_window_indices]
            
            # Count changes between consecutive messages
            changes = np.sum(np.any(sorted_payloads[1:] != sorted_payloads[:-1], axis=1))
            payload_change_rate[i] = changes / (len(window_indices) - 1)
        else:
            # If only one message, set entropy to that message's entropy and change rate to 0
            payload_entropy_last_20ms[i] = payload_entropy[window_indices[0]] if len(window_indices) == 1 else 0
            payload_change_rate[i] = 0

        end_time_change_payload = time.time()
        payload_change_timing.append(end_time_change_payload - start_time_change_payload)
    
    # Calculate averages
    average_time_id = sum(id_timing) / len(id_timing)
    average_time_average_payload = sum(payload_average_timing) / len(payload_average_timing)
    average_time_change_payload = sum(payload_change_timing) / len(payload_change_timing)
    average_time_payload_id = sum(payload_id_timing) / len(payload_id_timing)

    print(f"Average time for ID calculations: {average_time_id:.6f} seconds")
    print(f"Average time for payload AVERAGE calculations: {average_time_average_payload:.6f} seconds")
    print(f"Average time for payload CHANGE calculations: {average_time_change_payload:.6f} seconds")
    print(f"Average time for Entropy per ID calculations: {average_time_payload_id:.6f} seconds")

    # Add the result to the dataframe
    dataframe["msg_frequency"] = msg_count_last_20ms
    dataframe["entropy_average"] = payload_entropy_last_20ms
    dataframe["entropy_bit_change"] = payload_change_rate
    dataframe["entropy_id"] = id_specific_payload_entropy
    dataframe["entropy_id_std"] = id_specific_payload_entropy_std
    dataframe["entropy_id_z_score"] = id_specific_payload_entropy_z_score



    """
    # Normalize msg_count_last_50ms to [0, 1]
    max_count = dataframe["msg_frequency"].max()
    max_count_entropy_average = dataframe["entropy_average"].max()
    max_count_entropy_bit_change = dataframe["entropy_bit_change"].max()
    max_count_entropy_id_entropy = dataframe["entropy_id"].max()
    max_count_entropy_id_entropy_std = dataframe["entropy_id_std"].max()

    if max_count > 0:
        dataframe["msg_frequency"] = dataframe["msg_frequency"] / max_count
        
    else:
        dataframe["msg_frequency"] = 0  # If max_count is 0, set all normalized values to 0

    # Normalize entropy_average to [0, 1]
    if max_count_entropy_average > 0:
        dataframe["entropy_average"] = dataframe["entropy_average"] / max_count_entropy_average
    else:
        print("Max for entropy_average is 0 ")
        dataframe["entropy_average"] = 0  # If max_count_entropy_average is 0, set all normalized values to 0

    # Normalize entropy_bit_change to [0, 1]
    if max_count_entropy_bit_change > 0:
        dataframe["entropy_bit_change"] = dataframe["entropy_bit_change"] / max_count_entropy_bit_change
    else:
       dataframe["entropy_bit_change"] = 0  # If max_count_entropy_bit_change is 0, set all normalized values to 0

    # Normalize entropy_id_entropy to [0, 1]
    if max_count_entropy_id_entropy > 0:
        dataframe["entropy_id"] = dataframe["entropy_id"] / max_count_entropy_id_entropy
    else:
        print("Max for entropy_id is 0")
        dataframe["entropy_id"] = 0  # If max_count_entropy_id_entropy is 0, set all normalized values to 0

    # Normalize entropy_id_entropy_std to [0, 1]
    if max_count_entropy_id_entropy_std > 0:
        dataframe["entropy_id_std"] = dataframe["entropy_id_std"] / max_count_entropy_id_entropy_std
    else:
        print("Max for entropy_id_std is 0")
        dataframe["entropy_id_std"] = 0  # If max_count_entropy_id_entropy_std is 0, set all normalized values to 0
    """
    # Check for NaN values in 'id' or 'time' columns and handle
    if dataframe['arbitration_id'].isna().sum() > 0 or dataframe['timestamp'].isna().sum() > 0:
        print("NaN values found in 'id' or 'time' columns")
        dataframe = dataframe.dropna(subset=['arbitration_id', 'timestamp'])

    print(f"Compute Temporal Features completed {time.time() - start_time:.2f} seconds")

    return dataframe


def feature_selection_preparation(file_name, phase ,pre_dataframe = None, rows = None, binary = False, binary_id = True, embedding_model = None, id_to_embedding = None, scalers = None):
    start_time = time.time()
    print("#############START#####################")

    column_names_train = ['timestamp' , 'arbitration_id' , 'channel' , 'dlc', 'data' , 'ecu']
    column_names_test = ['timestamp', 'arbitration_id', 'dlc', 'data']
    # Define consistent dtypes for all phases
    # First check original file data types (read a few rows)
    """
    print(f"Checking original data types in {file_name}...")
    try:
        sample_df = pd.read_csv(file_name, nrows=1)
        #print("Original inferred data types:")
        #for col in sample_df.columns:
        #    print(f"  {col}: {sample_df[col].dtype}")

        # To see the actual value format
        print("\nSample values for each column:")
        for col in sample_df.columns:
            print(f"  {col}: {sample_df[col].iloc[0]} (type: {type(sample_df[col].iloc[0]).__name__})")
    except Exception as e:
        print(f"Error sampling file: {e}")
    """
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
            
        else:
            print("Invalid phase")
            return None
            
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

    dataframe = dataframe[dataframe['dlc'] == 8].reset_index(drop=True)
    dataframe['timestamp'] = dataframe['timestamp'] - dataframe['timestamp'].min()

    # Print the count of anomalies
    if phase == 'test':
        count_of_ones = dataframe['type'].sum()
        count_of_zeros = len(dataframe) - count_of_ones
        print(f"Normal entries in 'type' column : {count_of_zeros}")
        print(f"Anomalies in 'type' column: {count_of_ones}")

    ############# Extract data to indiviual bits or normalized integers ########################

    # Extract data to normalized integers 
    if not binary: 
        data_columns = [f'data[{i}]' for i in range(8)]
        dataframe[data_columns] = dataframe['data'].str.split(' ', expand=True).iloc[:, :8]
        
        # Convert Data from Hexadecimal to Integers [0,255]
        for col in data_columns:
            dataframe[col] = dataframe[col].apply(lambda x: int(x, 16) if isinstance(x, str) else x)

    # Extract data to bits        
    else:
        data_columns = [f'data[{i}]' for i in range(8)]
        dataframe[data_columns] = dataframe['data'].str.split(' ', expand=True).iloc[:, :8]

        # Converts each byte (hex value) to individual bits
        bit_columns = []
        for col in data_columns:
            # First convert hex to binary string
            dataframe[col] = dataframe[col].apply(lambda x: bin(int(x, 16))[2:].zfill(8) if isinstance(x, str) else x)
            
            # Binary string into individual bits
            for bit_pos in range(8):
                bit_col_name = f'{col}_data_bit_{bit_pos}'
                dataframe[bit_col_name] = dataframe[col].apply(lambda x: int(x[bit_pos]) if isinstance(x, str) and len(x) > bit_pos else None)
                bit_columns.append(bit_col_name)
    

    #######################################################################  
    # Entropy feature
    if not binary:
        dataframe['payload_entropy'] = dataframe.apply(calculate_entropy, axis=1)    

    ############# Extract Arbitration ID to Individual Bits or Embeddings ########################

    unique_ids_amount = dataframe['arbitration_id'].nunique()
    print(f"Amount of unique IDS in {phase}: "
    f"{unique_ids_amount}")

    if binary_id:
        num_bits = 29  # Standard for CAN IDs
        binary_encoded_ids = binary_encode_integers(dataframe['arbitration_id'].tolist(), num_bits)
        binary_encoded_df = pd.DataFrame(binary_encoded_ids, columns=[f'bit_{i}' for i in range(num_bits)])

        # Add binary-encoded IDs to the original DataFrame
        dataframe = pd.concat([binary_encoded_df, dataframe], axis=1)

        # Extract all columns at once as NumPy arrays
        bit_cols = dataframe[[f'bit_{i}' for i in range(num_bits)]].to_numpy()
    elif phase == 'training' and not binary_id: 
        embedding_dim = int(np.ceil(np.sqrt(unique_ids_amount)))

        unique_arbitration_ids = dataframe['arbitration_id'].unique()
        id_embeddings , embedding_model = train_embedding(unique_ids = unique_arbitration_ids,num_unique_ids= unique_ids_amount, labels= None, num_epochs= 100, embedding_dim= embedding_dim)

        # Create a default embedding (e.g., a zero vector)
        default_embedding = np.full(embedding_dim, -5)
        
        # Create a defaultdict with a default value (default_embedding)
        id_to_embedding = defaultdict(lambda: default_embedding, 
                                    {id: embedding for id, embedding in zip(unique_arbitration_ids, id_embeddings)})
        
        # Add embeddings to the DataFrame based on arbitration_id
        dataframe[[f'id_embed{i}' for i in range(embedding_dim)]] = dataframe['arbitration_id'].map(id_to_embedding).tolist()
        
        # Optional: verify the columns were added correctly
        bit_cols = dataframe[[f'id_embed{i}' for i in range(embedding_dim)]].to_numpy()
        
    else:
        # Extract embedding dimension from first entry in id_to_embedding
        first_item = next(iter(id_to_embedding.items()))
        embedding_dim = len(first_item[1])  # Get embedding size

        default_embedding = np.zeros(embedding_dim)


        # Add embeddings to the DataFrame based on arbitration_id
        dataframe[[f'id_embed{i}' for i in range(embedding_dim)]] = dataframe['arbitration_id'].map(id_to_embedding).tolist()
        
        # Optional: verify the columns were added correctly
        bit_cols = dataframe[[f'id_embed{i}' for i in range(embedding_dim)]].to_numpy()

    

    #######################################################################  

    # Compute Temporal Features and Normalize
    if not binary: 
        dataframe = compute_temporal_features(dataframe)
        if phase == "training":
            scalers = {}
            z_scaler_data_features = StandardScaler()
            # Apply Z-score normalization using StandardScaler
            z_scaler_temporal_features = StandardScaler()

            temporal_feature_columns = [
                "msg_frequency",
                "entropy_average",
                "entropy_bit_change",
                "entropy_id",
                "entropy_id_std",
                "timestamp",
                "payload_entropy"
            ]

            dataframe[temporal_feature_columns] = z_scaler_temporal_features.fit_transform(dataframe[temporal_feature_columns])
            dataframe[data_columns] = z_scaler_data_features.fit_transform(dataframe[data_columns])

            scalers['temporal_feature_scaler'] = z_scaler_temporal_features
            scalers['data_scaler'] = z_scaler_data_features

        else:

            z_scaler_temporal_features = scalers['temporal_feature_scaler'] 
            z_scaler_data_features = scalers['data_scaler'] 

            temporal_feature_columns = [
                "msg_frequency",
                "entropy_average",
                "entropy_bit_change",
                "entropy_id",
                "entropy_id_std",
                "timestamp",
                "payload_entropy"
            ]


            dataframe[temporal_feature_columns] = z_scaler_temporal_features.transform(dataframe[temporal_feature_columns])
            dataframe[data_columns] = z_scaler_data_features.transform(dataframe[data_columns])


            

        #### Min-Max Scaler ###
        """
        dataframe[data_columns] = scaler.fit_transform(dataframe[data_columns])
        dataframe['timestamp'] = scaler.fit_transform(dataframe[['timestamp']])
        dataframe['payload_entropy'] = scaler.fit_transform(dataframe[['payload_entropy']])
        """

        #### Z-Score Scaler ###
        #dataframe[data_columns] = z_scaler.fit_transform(dataframe[data_columns])
        #dataframe['timestamp'] = z_scaler.fit_transform(dataframe[['timestamp']])
        #dataframe['payload_entropy'] = z_scaler.fit_transform(dataframe[['payload_entropy']]) # return scaler and use same on test



    # Concatenate along axis=1
    if not binary: 
        data_cols = dataframe[data_columns].to_numpy()
        extra_cols = dataframe[['msg_frequency', 'timestamp', 'payload_entropy',
                            'entropy_average', 'entropy_id', 'entropy_id_std']].to_numpy()
        dataframe['features'] = list(np.hstack([bit_cols, data_cols, extra_cols]))
        #dataframe['features'] = list(np.hstack([data_cols, extra_cols]))
        #dataframe['features'] = list(extra_cols)
    else:
        # Step 3: Collect bit-wise columns and store as a list
        bit_data_cols = [col for col in dataframe.columns if '_data_bit_' in col]
        data_cols = dataframe[bit_data_cols].to_numpy()
        dataframe['features'] = list(np.hstack([bit_cols, data_cols]))
     
        #[row[[f'bit_{i}' for i in range(num_bits)]].values,  # Binary data columns 0 to 28
        # row[data_columns].values,  # Other data columns 28 to 36
        #np.array([row['msg_frequency'],row['timestamp'],row['payload_entropy'],row['entropy_average'], row['entropy_bit_change']
        #          , row['entropy_id'], row['entropy_id_std'], row['entropy_id_z_score']])  # Additional features
        #]), axis=1)

       
    nan_counts = dataframe.isna().sum()
    if nan_counts.any():
        print("NaN values found:\n", nan_counts[nan_counts > 0])

        # Identify and print the rows with NaN values
        nan_rows = dataframe[dataframe.isna().any(axis=1)]
        print("\nRows containing NaN values:\n", nan_rows)
    
    if np.any(np.isinf(dataframe.select_dtypes(include=[np.number]))):
        print("Dataframe includes INF values")
    print(f"Feature Selection completed in {time.time() - start_time:.2f} seconds")

    if phase == 'test':
        return dataframe
    return dataframe, embedding_model, id_to_embedding, scalers


def create_sliding_windows(data, labels=None, window_size=5, stride=1, anomaly_window = 5):
    start_time = time.time()
    # Generates sliding windows for both features and labels.
    X = np.array([data[i:i+window_size] for i in range(0, len(data) - window_size + 1, stride)], dtype=np.float32)
    max_anomaly = 0
    if labels is not None:
        labels = labels.values

        # Initialize an empty list to store the labels
        y = []

        # Define threshold: At least 50% of the window should contain 1s
        threshold = anomaly_window 
        for i in range(0, len(labels) - window_size + 1, stride):
            # Extract the current window of labels
            window = labels[i:i+window_size]
            # Check if there is at least one '1' in the window
            amount_anomaly = np.sum(window == 1) 
            print("Window Anomaly Amount ", amount_anomaly)
            #print(f"index: {i} and {i + window_size}")
            if amount_anomaly > max_anomaly:
                max_anomaly = amount_anomaly

            if amount_anomaly >= threshold:  # Count 1s and compare to threshold
                y.append(1)
            else:
                y.append(0)  # If there's not enough 1s in window, mark this window as normal

        # Convert the list to a numpy array
        y = np.array(y, dtype=np.float32)
        y_len = len(y)
        count_of_ones = np.sum(y)
        
        print("Max Amount of Anomalies Found in a Window: ", max_anomaly)
        print(f"Sliding Window (test) completed in {time.time() - start_time:.7f} seconds")
        print("-----------------------------------")
        print("Original window that works: ", X.shape)  # (991, 50, 40)
        print(f"Max Amount of anomalies Allowed per window: ", threshold)
        print(f"Normals in 'y' array: {y_len - count_of_ones}")
        print(f"Anomalies in 'y' array: {count_of_ones}")


        return X, y
    print(f"Sliding completed (train) in {time.time() - start_time:.7f} seconds")
    print("-----------------------------------")
    print("Original window that works: ", X.shape)  # (991, 50, 40)

    return X



def convert_to_tensorflow(featureframe, labels=None, batch_size=32, window_size=5, stride=1, split_ratio = 0.8, window_anomaly = 5):
    # Convert feature list to NumPy array
    input_data = np.array(featureframe.tolist(), dtype=np.float32)
    before_window_shape = input_data.shape

    # Check if Train or Test dataframe
    if labels is not None:
        input_data, labels = create_sliding_windows(input_data, labels, window_size, stride, window_anomaly)
        #print("x before tensor", len(input_data))
        #print("y before tensor", len(labels))
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
        print("#####################################")
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

        # Split val_dataset again
        val_size = dataset_size - train_size
        val_split_size = val_size // 2  # half
        
        val_dataset_1 = val_dataset.take(val_split_size)
        val_dataset_2 = val_dataset.skip(val_split_size)

        # Batch
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset1 = val_dataset_1.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset2 = val_dataset_2.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        print(f"Feature shape BEFORE sliding window: {before_window_shape}")
        print(f"Feature shape AFTER sliding window: {input_data.shape}")
        print("#####################################")
        return train_dataset, val_dataset1, val_dataset2

    # Apply batching
    model_input = model_input.batch(batch_size)
    print(f"Feature shape BEFORE sliding window: {before_window_shape}")
    print(f"Feature shape AFTER sliding window: {input_data.shape}")
    print(f"Successfully prepared model input data.")
    return model_input


######################################

def feature_selection_preparation_new(file_name, phase ,pre_dataframe = None, rows = None, binary = False, binary_id = True, embedding_model = None, id_to_embedding = None, scalers = None, train_embedding_scaler = False):
    start_time = time.time()
    print("#############START#####################")

    column_names_train = ['timestamp' , 'arbitration_id' , 'channel' , 'dlc', 'data' , 'ecu']
    column_names_test = ['timestamp', 'arbitration_id', 'dlc', 'data']
    # Define consistent dtypes for all phases
    # First check original file data types (read a few rows)
    """
    print(f"Checking original data types in {file_name}...")
    try:
        sample_df = pd.read_csv(file_name, nrows=1)
        #print("Original inferred data types:")
        #for col in sample_df.columns:
        #    print(f"  {col}: {sample_df[col].dtype}")

        # To see the actual value format
        print("\nSample values for each column:")
        for col in sample_df.columns:
            print(f"  {col}: {sample_df[col].iloc[0]} (type: {type(sample_df[col].iloc[0]).__name__})")
    except Exception as e:
        print(f"Error sampling file: {e}")
    """
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
            
        else:
            print("Invalid phase")
            return None
            
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

    
    dataframe = dataframe[dataframe['dlc'] == 8].reset_index(drop=True)
    dataframe['timestamp'] = dataframe['timestamp'] - dataframe['timestamp'].min()

    # Print the count of anomalies
    if phase == 'test':
        count_of_ones = dataframe['type'].sum()
        count_of_zeros = len(dataframe) - count_of_ones
        print(f"Normal entries in 'type' column : {count_of_zeros}")
        print(f"Anomalies in 'type' column: {count_of_ones}")

    ############# Extract data to indiviual bits or normalized integers ########################

    # Extract data to normalized integers 
    if not binary: 
        data_columns = [f'data[{i}]' for i in range(8)]
        dataframe[data_columns] = dataframe['data'].str.split(' ', expand=True).iloc[:, :8]
        
        # Convert Data from Hexadecimal to Integers [0,255]
        for col in data_columns:
            dataframe[col] = dataframe[col].apply(lambda x: int(x, 16) if isinstance(x, str) else x)

    # Extract data to bits        
    else:
        data_columns = [f'data[{i}]' for i in range(8)]
        dataframe[data_columns] = dataframe['data'].str.split(' ', expand=True).iloc[:, :8]

        # Converts each byte (hex value) to individual bits
        bit_columns = []
        for col in data_columns:
            # First convert hex to binary string
            dataframe[col] = dataframe[col].apply(lambda x: bin(int(x, 16))[2:].zfill(8) if isinstance(x, str) else x)
            
            # Binary string into individual bits
            for bit_pos in range(8):
                bit_col_name = f'{col}_data_bit_{bit_pos}'
                dataframe[bit_col_name] = dataframe[col].apply(lambda x: int(x[bit_pos]) if isinstance(x, str) and len(x) > bit_pos else None)
                bit_columns.append(bit_col_name)
    

    #######################################################################  
    # Entropy feature
    if not binary:
        dataframe['payload_entropy'] = dataframe.apply(calculate_entropy, axis=1)    

    ############# Extract Arbitration ID to Individual Bits or Embeddings ########################

    unique_ids_amount = dataframe['arbitration_id'].nunique()
    print(f"Amount of unique IDS in {phase}: "
    f"{unique_ids_amount}")

    if binary_id:
        num_bits = 29  # Standard for CAN IDs
        binary_encoded_ids = binary_encode_integers(dataframe['arbitration_id'].tolist(), num_bits)
        binary_encoded_df = pd.DataFrame(binary_encoded_ids, columns=[f'bit_{i}' for i in range(num_bits)])

        # Add binary-encoded IDs to the original DataFrame
        dataframe = pd.concat([binary_encoded_df, dataframe], axis=1)

        # Extract all columns at once as NumPy arrays
        bit_cols = dataframe[[f'bit_{i}' for i in range(num_bits)]].to_numpy()

    #elif phase == 'training' and not binary_id: 
    elif  id_to_embedding is None and not binary_id: 
        embedding_dim = int(np.ceil(np.sqrt(unique_ids_amount)))

        unique_arbitration_ids = dataframe['arbitration_id'].unique()
        id_embeddings , embedding_model = train_embedding(unique_ids = unique_arbitration_ids,num_unique_ids= unique_ids_amount, labels= None, num_epochs= 100, embedding_dim= embedding_dim)

        # Create a default embedding (e.g., a zero vector)
        default_embedding = np.full(embedding_dim, -5)
        
        # Create a defaultdict with a default value (default_embedding)
        id_to_embedding = defaultdict(lambda: default_embedding, 
                                    {id: embedding for id, embedding in zip(unique_arbitration_ids, id_embeddings)})
        
        # Add embeddings to the DataFrame based on arbitration_id
        dataframe[[f'id_embed{i}' for i in range(embedding_dim)]] = dataframe['arbitration_id'].map(id_to_embedding).tolist()
        
        # Optional: verify the columns were added correctly
        bit_cols = dataframe[[f'id_embed{i}' for i in range(embedding_dim)]].to_numpy()
        
    else:
        # Extract embedding dimension from first entry in id_to_embedding
        first_item = next(iter(id_to_embedding.items()))
        embedding_dim = len(first_item[1])  # Get embedding size

        default_embedding = np.zeros(embedding_dim)


        # Add embeddings to the DataFrame based on arbitration_id
        dataframe[[f'id_embed{i}' for i in range(embedding_dim)]] = dataframe['arbitration_id'].map(id_to_embedding).tolist()
        
        # Optional: verify the columns were added correctly
        bit_cols = dataframe[[f'id_embed{i}' for i in range(embedding_dim)]].to_numpy()


    #######################################################################  

    # Compute Temporal Features and Normalize
    if not binary: 
        dataframe = compute_temporal_features(dataframe)
        #if phase == "training": 
        if scalers is None:
            scalers = {}
            z_scaler_data_features = StandardScaler()
            # Apply Z-score normalization using StandardScaler
            z_scaler_temporal_features = StandardScaler()

            temporal_feature_columns = [
                "msg_frequency",
                "entropy_average",
                "entropy_bit_change",
                "entropy_id",
                "entropy_id_std",
                "timestamp",
                "payload_entropy"
            ]

            dataframe[temporal_feature_columns] = z_scaler_temporal_features.fit_transform(dataframe[temporal_feature_columns])
            dataframe[data_columns] = z_scaler_data_features.fit_transform(dataframe[data_columns])

            scalers['temporal_feature_scaler'] = z_scaler_temporal_features
            scalers['data_scaler'] = z_scaler_data_features

        else:

            z_scaler_temporal_features = scalers['temporal_feature_scaler'] 
            z_scaler_data_features = scalers['data_scaler'] 

            temporal_feature_columns = [
                "msg_frequency",
                "entropy_average",
                "entropy_bit_change",
                "entropy_id",
                "entropy_id_std",
                "timestamp",
                "payload_entropy"
            ]


            dataframe[temporal_feature_columns] = z_scaler_temporal_features.transform(dataframe[temporal_feature_columns])
            dataframe[data_columns] = z_scaler_data_features.transform(dataframe[data_columns])


            

        #### Min-Max Scaler ###
        """
        dataframe[data_columns] = scaler.fit_transform(dataframe[data_columns])
        dataframe['timestamp'] = scaler.fit_transform(dataframe[['timestamp']])
        dataframe['payload_entropy'] = scaler.fit_transform(dataframe[['payload_entropy']])
        """

        #### Z-Score Scaler ###
        #dataframe[data_columns] = z_scaler.fit_transform(dataframe[data_columns])
        #dataframe['timestamp'] = z_scaler.fit_transform(dataframe[['timestamp']])
        #dataframe['payload_entropy'] = z_scaler.fit_transform(dataframe[['payload_entropy']]) # return scaler and use same on test



    # Concatenate along axis=1
    if not binary: 
        data_cols = dataframe[data_columns].to_numpy()
        extra_cols = dataframe[['msg_frequency', 'timestamp', 'payload_entropy',
                            'entropy_average', 'entropy_id', 'entropy_id_std']].to_numpy()
        dataframe['features'] = list(np.hstack([bit_cols, data_cols, extra_cols]))
        #dataframe['features'] = list(np.hstack([data_cols, extra_cols]))
        #dataframe['features'] = list(extra_cols)
    else:
        # Step 3: Collect bit-wise columns and store as a list
        bit_data_cols = [col for col in dataframe.columns if '_data_bit_' in col]
        data_cols = dataframe[bit_data_cols].to_numpy()
        dataframe['features'] = list(np.hstack([bit_cols, data_cols]))
     
        #[row[[f'bit_{i}' for i in range(num_bits)]].values,  # Binary data columns 0 to 28
        # row[data_columns].values,  # Other data columns 28 to 36
        #np.array([row['msg_frequency'],row['timestamp'],row['payload_entropy'],row['entropy_average'], row['entropy_bit_change']
        #          , row['entropy_id'], row['entropy_id_std'], row['entropy_id_z_score']])  # Additional features
        #]), axis=1)

       
    nan_counts = dataframe.isna().sum()
    if nan_counts.any():
        print("NaN values found:\n", nan_counts[nan_counts > 0])

        # Identify and print the rows with NaN values
        nan_rows = dataframe[dataframe.isna().any(axis=1)]
        print("\nRows containing NaN values:\n", nan_rows)
    
    if np.any(np.isinf(dataframe.select_dtypes(include=[np.number]))):
        print("Dataframe includes INF values")
    print(f"Feature Selection completed in {time.time() - start_time:.2f} seconds")

    #if phase == 'test':
    if not train_embedding_scaler:   
        print("returning only df")   
        return dataframe
    
    return dataframe, embedding_model, id_to_embedding, scalers
