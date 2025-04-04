import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, auc, roc_curve, f1_score
import time
from utils import get_confusion_matrix
from sklearn.neighbors import BallTree
import hdbscan
from hdbscan.prediction import approximate_predict
from sklearn.manifold import TSNE
from train import compute_loss_binary, compute_loss_continous, compute_loss_binary_continous


"""
The following functions handles clustering AD
"""
def prepare_features(means_list, variances_list):
    """
    Combine and scale mean and variance features.
    
    Args:
        means_list (np.ndarray): Array of shape (n_samples, 20) containing mean values
        variances_list (np.ndarray): Array of shape (n_samples, 20) containing variance values
        
    Returns:
        tuple: (scaled_features, scaler)
    """
    # Combine mean and variance features
    features = np.hstack((means_list, variances_list))
    
    # Log transform variances (optional but often helpful as variances can be skewed)
    # Adding a small constant to avoid log(0)
    #features[:, means_list.shape[1]:] = np.log(features[:, means_list.shape[1]:] + 1e-10)
    #features[:, means_list.shape[1]:] = np.log(features[:, means_list.shape[1]:] + 1 + 1e-10)
    # Scale the features
    scaler = StandardScaler()
    #scaled_features = scaler.fit_transform(features)
    
    return features, scaler

def train_isolation_forest(normal_features, contamination=0.01, n_estimators=100, random_state=42):
    """
    Train an Isolation Forest model for anomaly detection.
    
    Args:
        normal_features (np.ndarray): Features from normal data only
        contamination (float): Expected proportion of outliers in training data
        n_estimators (int): Number of base estimators
        random_state (int): Random seed
        
    Returns:
        IsolationForest: Trained model
    """
    #print("----Training Isolation Forest----")
    start_time = time.time()
    
    # Initialize and fit Isolation Forest
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    
    model.fit(normal_features)
    
    print(f"Isolation Forest training completed in {time.time() - start_time:.4f} seconds")
    return model

def train_one_class_svm(normal_features, nu=0.01, kernel='rbf', gamma='scale'):
    """
    Train a One-Class SVM model for anomaly detection.
    
    Args:
        normal_features (np.ndarray): Features from normal data only
        nu (float): Upper bound on the fraction of training errors
        kernel (str): Kernel type ('rbf', 'linear', 'poly')
        gamma (str or float): Kernel coefficient
        
    Returns:
        OneClassSVM: Trained model
    """
    #print("----Training One-Class SVM----")
    start_time = time.time()
    
    # Initialize and fit One-Class SVM
    model = OneClassSVM(
        nu=nu,
        kernel=kernel,
        gamma=gamma
    )
    
    model.fit(normal_features)
    
    print(f"One-Class SVM training completed in {time.time() - start_time:.4f} seconds")
    return model

def train_hdbscan_detector(normal_features, min_cluster_size=5, min_samples=None, 
                           cluster_selection_epsilon=0.0, **kwargs):
    """
    Train an HDBSCAN model on normal data only.
    
    Args:
        normal_features (np.ndarray): Features from normal data only
        min_cluster_size (int): The minimum size of clusters
        min_samples (int): The number of samples in a neighborhood for a point to be a core point
        cluster_selection_epsilon (float): The distance threshold for cluster merging
        **kwargs: Additional parameters for HDBSCAN
        
    Returns:
        object: Trained HDBSCAN clusterer
    """
    start_time = time.time()
    # Initialize and fit HDBSCAN on normal data
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=True  # Add this parameter
    )
    clusterer.fit(normal_features)
    
    print(f"HDBSCAN training completed in {time.time() - start_time:.4f} seconds")

    return clusterer

def hdbscan_set_threshold(clusterer, normal_features, strictness=0.99):
    """
    Set an anomaly threshold based on clean normal data (no anomalies).
    
    Args:
        clusterer: Trained HDBSCAN model
        normal_features: Features from normal data (assumed to be clean)
        strictness: How strict to make the threshold (0.99 = include 99% of normal data)
        
    Returns:
        float: Threshold value for outlier scores
    """
    start_time = time.time()
    # Get outlier scores for normal data
    _, strengths = hdbscan.approximate_predict(clusterer, normal_features)
    outlier_scores = 1 - strengths  # Convert from cluster membership strength to outlier score
    
    # Set threshold to include the specified percentage of normal data
    threshold = np.percentile(outlier_scores, 100 * strictness)
    
    print(f"Threshold set at {threshold} (based on {strictness*100}% strictness)")
    print(f"This threshold considers the top {(1-strictness)*100:.2f}% of normal data as anomalous")
    
    # Validate threshold on training data
    anomaly_mask = outlier_scores >= threshold
    print(f"Detected {sum(anomaly_mask)} data points above threshold out of {len(normal_features)} samples")
    print(f"Percentage above threshold: {sum(anomaly_mask) / len(normal_features) * 100:.2f}%")
    print(f"HDBSCAN Threshold tuning completed in {time.time() - start_time:.4f} seconds")

    return threshold
    
def detect_anomalies_isolation_forest(model, features):
    """
    Detect anomalies using Isolation Forest.
    
    Args:
        model (IsolationForest): Trained Isolation Forest model
        features (np.ndarray): Features to detect anomalies in
        
    Returns:
        tuple: (anomaly_mask, anomaly_scores)
    """
    print("----Isolation Forest AD----")

    start_time = time.time()
    # Predict raw anomaly scores
    # Lower (more negative) scores indicate anomalies
    raw_scores = model.score_samples(features)
    
    # Predict labels (1: normal, -1: anomaly)
    predictions = model.predict(features)
    anomaly_mask = predictions == -1
    #anomaly_mask = predictions = (predictions == -1).astype(int)  # 1 for anomaly, 0 for normal
    
    # Convert raw scores to anomaly scores (higher = more anomalous)
    anomaly_scores = -raw_scores
    print(f"Anomali Detection using Isolation Forest completed in : {time.time() - start_time:.4f} seconds")
    print(f"Detected {np.sum(anomaly_mask)} anomalies out of {len(features)} samples")
    print(f"Anomaly percentage: {np.sum(anomaly_mask)/len(features)*100:.2f}%")
    print("######################")

    return anomaly_mask, anomaly_scores

def detect_anomalies_one_class_svm(model, features):
    """
    Detect anomalies using One-Class SVM.
    
    Args:
        model (OneClassSVM): Trained One-Class SVM model
        features (np.ndarray): Features to detect anomalies in
        
    Returns:
        tuple: (anomaly_mask, anomaly_scores)
    """
    print("----1-Class SVM AD----")

    start_time = time.time()
    # Predict raw decision function values
    # Lower (more negative) values indicate anomalies
    raw_scores = model.decision_function(features)
    
    # Predict labels (1: normal, -1: anomaly)
    predictions = model.predict(features)
    anomaly_mask = predictions == -1
    #anomaly_mask = predictions = (predictions == -1).astype(int)  # 1 for anomaly, 0 for normal
    
    # Convert raw scores to anomaly scores (higher = more anomalous)
    anomaly_scores = -raw_scores
    print(f"Anomali Detection using 1-Class SVM : {time.time() - start_time:.4f} seconds")
    print(f"Detected {np.sum(anomaly_mask)} anomalies out of {len(features)} samples")
    print(f"Anomaly percentage: {np.sum(anomaly_mask)/len(features)*100:.2f}%")
    print("######################")

    return anomaly_mask, anomaly_scores

def detect_anomalies_one_class_svm_with_threshold(model, features, true_labels, desired_threshold=0.9):
    """
    Detect anomalies using One-Class SVM with threshold tuning based on precision and recall.

    Args:
        model (OneClassSVM): Trained One-Class SVM model
        features (np.ndarray): Features to detect anomalies in
        true_labels (np.ndarray): True labels to help determine the optimal threshold
        desired_threshold (float): Threshold value between 0 and 1 to optimize precision-recall trade-off

    Returns:
        tuple: (anomaly_mask, anomaly_scores)
    """
    print("----1-Class SVM w/ Tuning AD----")

    start_time = time.time()
    # Predict raw decision function values
    raw_scores = model.decision_function(features)
    
    # Calculate precision, recall, and thresholds based on true labels
    precision, recall, thresholds = precision_recall_curve(true_labels, -raw_scores)  # Negative because higher scores mean more anomalous
    raw_scores_neg = raw_scores

    
    # Make sure raw_scores and true_labels are numpy arrays
    raw_scores_neg = -np.array(raw_scores)
    true_labels = np.array(true_labels)

    # Create histograms to visualize the raw scores
    plt.figure(figsize=(10, 6))

    # Plot histogram for positive class (true label = 1)
    plt.hist(raw_scores_neg[true_labels == 1], bins=30, alpha=0.7, color='blue', label='True Label = 1')

    # Plot histogram for negative class (true label = 0)
    plt.hist(raw_scores_neg[true_labels == 0], bins=30, alpha=0.7, color='red', label='True Label = 0')

    # Labels and title
    plt.xlabel('Raw Scores (negated)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Raw Scores by True Labels')

    # Adding a legend
    plt.legend()

    # Show the plot
    plt.show()

    # Find the threshold that achieves the desired recall or precision
    # Select the threshold corresponding to the desired recall or precision
    # Calculate F1 scores for each threshold
    # Apply mask before F1 score calculation to avoid NaN
    valid_indices = (precision > 0) & (recall > 0)

    # Calculate F1 scores, but only for valid indices
    f1_scores = 2 * (precision[valid_indices] * recall[valid_indices]) / (precision[valid_indices] + recall[valid_indices])

    # Find the threshold corresponding to the maximum F1 score
    threshold_index = np.argmax(f1_scores)
    #threshold_index = np.argmax(recall)
    threshold = thresholds[threshold_index]
    print("Threshold based on F1: ", threshold)
    print("F1: ", f1_scores[threshold_index])
    print("Recall based on F1: ", recall[threshold_index])
    print("Precision based on F1: ", precision[threshold_index])
    # Classify as anomalies based on the selected threshold
    anomaly_mask = -raw_scores >= threshold  # Anomalies are the points where the decision function is below the threshold

    # Convert raw scores to anomaly scores (higher = more anomalous)
    anomaly_scores = -raw_scores
    print(f"Anomali Detection using 1-Class SVM with Threshold Tuning completed in : {time.time() - start_time:.4f} seconds")
    print(f"Detected {np.sum(anomaly_mask)} anomalies out of {len(features)} samples")
    print(f"Anomaly percentage: {np.sum(anomaly_mask) / len(features) * 100:.2f}%")
    print("######################")

    return anomaly_mask, anomaly_scores

def detect_anomalies_hdbscan(clusterer, features, threshold = 0.9):
    """
    Detect anomalies using the trained HDBSCAN model without a fixed threshold.
   
    Args:
        clusterer: Trained HDBSCAN model
        features (np.ndarray): Features to detect anomalies in
       
    Returns:
        tuple: (anomaly_mask, anomaly_scores)
    """


    test_labels, strengths = approximate_predict(clusterer, features)
    outliers = strengths < threshold
    #outliers = 1 - strengths
    return outliers, test_labels, strengths

def visualize_anomalies(features, anomaly_mask, title="Anomaly Detection Results", 
                            perplexity=30, n_iter=1000):
    """
    Visualize detected anomalies in 2D using t-SNE.
    
    Args:
        features (np.ndarray): Feature array
        anomaly_mask (np.ndarray): Boolean array indicating anomalies
        title (str): Plot title
        perplexity (int): t-SNE perplexity parameter
        n_iter (int): Number of iterations for t-SNE
    """
    
    # Reduce to 2D for visualization using t-SNE
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    
    # Plot normal points
    plt.scatter(
        features_2d[~anomaly_mask, 0], 
        features_2d[~anomaly_mask, 1], 
        c='blue', alpha=0.6, label='Normal'
    )
    
    # Plot anomalies
    plt.scatter(
        features_2d[anomaly_mask, 0], 
        features_2d[anomaly_mask, 1], 
        c='red', marker='x', s=100, label='Anomalies'
    )
    
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Step 4: Visualize results (optional)
def visualize_results(normal_latent, test_latent, test_outliers, true_labels=None):
    # Combine data for TSNE
    combined_latent = np.vstack([normal_latent, test_latent])
    
    # Apply TSNE
    tsne = TSNE(n_components=2, random_state=42)
    combined_tsne = tsne.fit_transform(combined_latent)
    
    # Split back
    normal_tsne = combined_tsne[:len(normal_latent)]
    test_tsne = combined_tsne[len(normal_latent):]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(normal_tsne[:, 0], normal_tsne[:, 1], c='blue', label='Normal Training Data')
    plt.scatter(test_tsne[~test_outliers, 0], test_tsne[~test_outliers, 1], c='green', label='Test Normal')
    plt.scatter(test_tsne[test_outliers, 0], test_tsne[test_outliers, 1], c='red', label='Detected Anomalies')
    
    if true_labels is not None:
        accuracy = np.mean((test_outliers == true_labels))
        plt.title(f'Anomaly Detection Results (Accuracy: {accuracy:.2f})')
    else:
        plt.title('Anomaly Detection Results')
    
    plt.legend()
    plt.show()

def evaluate_anomaly_detector(anomaly_scores, true_labels, model_name="Model"):
    """
    Evaluate anomaly detection performance using ROC and PR curves.
    
    Args:
        anomaly_scores (np.ndarray): Anomaly scores (higher = more anomalous)
        true_labels (np.ndarray): True binary labels (0=normal, 1=anomaly)
        model_name (str): Name of the model for plot labels
    """
    
    # Calculate PR curve and AUC
    precision, recall, _ = precision_recall_curve(true_labels, anomaly_scores)
    pr_auc = auc(recall, precision)

    
    # Plot ROC curve
    plt.figure(figsize=(12, 5))
    
    # Plot PR curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    
    plt.tight_layout()
    plt.show()

    print(f"{model_name} Precision: {precision}")
    print(f"{model_name} Recall: {recall}")
    print(f"{model_name} PR AUC: {pr_auc:.4f}")

def evaluate_hdbscan_detector(true_labels, predicted_anomalies):
    """
    Evaluate HDBSCAN anomaly detection performance.
    
    Args:
        true_labels (array-like): Ground truth labels where 1=attack, 0=normal
        predicted_anomalies (array-like): Predicted anomaly mask from detect_anomalies_hdbscan
        
    Returns:
        dict: Dictionary containing various performance metrics
    """
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
    from sklearn.metrics import recall_score, f1_score, roc_auc_score
    
    # Calculate basic metrics
    cm = confusion_matrix(true_labels, predicted_anomalies)
    
    # Extract values from confusion matrix
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"True Negatives: {tn} | False Positives: {fp}")
        print(f"False Negatives: {fn} | True Positives: {tp}")
    else:
        print("Warning: Confusion matrix is not 2x2, possible issue with labels")
    
    # Calculate and print metrics
    accuracy = accuracy_score(true_labels, predicted_anomalies)
    precision = precision_score(true_labels, predicted_anomalies, zero_division=0)
    recall = recall_score(true_labels, predicted_anomalies, zero_division=0)
    f1 = f1_score(true_labels, predicted_anomalies, zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (% of predicted anomalies that are actual attacks)")
    print(f"Recall: {recall:.4f} (% of actual attacks that were detected)")
    print(f"F1 Score: {f1:.4f}")
    
    # Calculate ROC AUC if anomaly scores are available
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    
    # Return all metrics in a dictionary
    return metrics




"""
The following functions computes reconstruction error and probability
"""

def compute_probability_continous(reconstructed, batch):
    #print("Reconstructed shape: ", reconstructed.shape)

    # Mean over Samples  (Monte Carlo approximation of expectation)
    #reconstructed = tf.reduce_mean(reconstructed, axis = 0)

    n_samples = reconstructed.shape[0]
    # Broadcast batch to match reconstructed's shape
    batch = tf.repeat(tf.expand_dims(batch, axis=0), n_samples, axis=0)

    # Compute log probabilities for continuous features (Gaussian)
    continuous_log_probs = -0.5 * tf.math.log(2 * np.pi) - \
                                  0.5 * tf.square(batch - reconstructed)
    #print("CONTINOUS log_probs shape: " , continuous_log_probs.shape)
        
     # Combine log probabilities
    log_probs = continuous_log_probs
    #print("log_probs shape: " , log_probs.shape)
    # Sum log probabilities across features (axis=-1)
    sample_log_probs = tf.reduce_sum(log_probs, axis=-1)
    #print("log_probs shape after summing over features: " , sample_log_probs.shape)
      
    # Average over Samples and Windows
    mean_log_prob = tf.reduce_mean(sample_log_probs, axis=(0, 2))
            
    # Convert to negative log-likelihood for anomaly score (lower probability = higher anomaly score)
    reconstruction_score = -mean_log_prob
    reconstruction_score.numpy()
    #print("mean final reconstruction score shape: " , reconstruction_score.shape)
    return reconstruction_score

def compute_probability_binary_continous(reconstructed, batch):
    print("Reconstructed shape: ", reconstructed.shape)

    # Mean over Samples  (Monte Carlo approximation of expectation)
    n_samples = reconstructed.shape[0]
    # Broadcast batch to match reconstructed's shape
    batch = tf.repeat(tf.expand_dims(batch, axis=0), n_samples, axis=0)
    binary_features = 29

    # Extract features
    batch_binary = batch[..., :binary_features]
    reconstructed_binary = reconstructed[..., :binary_features]
    batch_continuous = batch[..., binary_features:]
    reconstructed_continuous = reconstructed[..., binary_features:] 

    print("Reconstructed shape: ", reconstructed.shape)
    print("Batch shape: ", batch.shape)

    # Compute log probabilities for binary features (Bernoulli)
    binary_log_probs = batch_binary * tf.math.log(reconstructed_binary + 1e-10) + \
                        (1 - batch_binary) * tf.math.log(1 - reconstructed_binary + 1e-10)
    print("BINARY log_probs shape: " , binary_log_probs.shape)

    # Compute log probabilities for continuous features (Gaussian)
    # Assume diagonal covariance matrix with learned or fixed variance
    continuous_variance = 0.1  # Fixed variance, can be replaced with learned variance
    continuous_log_probs = -0.5 * tf.math.log(2 * np.pi * continuous_variance) - \
                                  0.5 * tf.square(batch_continuous - reconstructed_continuous) / continuous_variance
    print("CONTINOUS log_probs shape: " , continuous_log_probs.shape)
        
     # Combine log probabilities
    log_probs = tf.concat([binary_log_probs, continuous_log_probs], axis=-1)
    print("log_probs shape: " , log_probs.shape)
    # Sum log probabilities across features (axis=-1)
    sample_log_probs = tf.reduce_sum(log_probs, axis=-1)
            
    # Average over window
    mean_log_prob = tf.reduce_mean(sample_log_probs, axis=(0, 2))

            
    # Convert to negative log-likelihood for anomaly score (lower probability = higher anomaly score)
    reconstruction_score = -mean_log_prob
    reconstruction_score.numpy()
    print("mean final reconstruction score shape: " , reconstruction_score.shape)
    return reconstruction_score

def compute_probability_binary(reconstructed,batch):
    # Mean over Samples  (Monte Carlo approximation of expectation)
    # Mean over Samples  (Monte Carlo approximation of expectation)
    n_samples = reconstructed.shape[0]
    # Broadcast batch to match reconstructed's shape
    batch = tf.repeat(tf.expand_dims(batch, axis=0), n_samples, axis=0)


    # Compute log probabilities for binary features (Bernoulli)
    binary_log_probs = batch * tf.math.log(reconstructed + 1e-10) + \
                        (1 - batch) * tf.math.log(1 - reconstructed + 1e-10)

    log_probs = binary_log_probs
    # Sum log probabilities across features (axis=-1)
    sample_log_probs = tf.reduce_sum(log_probs, axis=-1)
            
    # Average over window
    mean_log_prob = tf.reduce_mean(sample_log_probs, axis=(0, 2))
            
    # Convert to negative log-likelihood for anomaly score (lower probability = higher anomaly score)
    reconstruction_score = -mean_log_prob
    reconstruction_score.numpy()
    return reconstruction_score

"""
The following functions retrieves Mean and Variances from a dataset
get_threshold_from_train() also returns latent threshold and tree structure based on the retrieved Mean And Variances
"""

def get_threshold_from_train(model_path, train_dataset, val_dataset,reconstruction_AD, latent_AD, binary = False):    
    start_time = time.time()
    load_vae = keras.models.load_model(model_path)
    load_vae.trainable = False  # Freeze model weights
    tree = None
    # Compute reconstruction losses on the test set with multiple samples
    n_samples = 10  # Number of latent samples during inference
    reconstruction_errors_threshold = []
    reconstruction_probs_threshold = []

    mean_train = []
    variance_train = []

    normal_distances_threshold = []
    if latent_AD:

        for batch in train_dataset:

            reconstructed, mu, logvar = load_vae(batch, n_samples=n_samples, latent_only = latent_AD)  
            batch_data = batch.numpy() 

            for i in range(len(batch)):

                mean_train.append(mu[i])
                variance_train.append(np.exp(logvar[i]))

        print(f"Collect Mean & Variance from Train completed in {time.time() - start_time:.4f} seconds")

        # Preprocess your normal data
        start_time_ball_tree = time.time()
        combined_params = np.hstack([mean_train, variance_train])  # Combine mean and variance
        tree = BallTree(combined_params, metric='pyfunc', func=bhattacharyya_distance_old_balltree)
        print(f"Ball Tree completed in {time.time() - start_time_ball_tree:.4f} seconds")

    debug = 0
    for batch in val_dataset:

        reconstructed, mu, logvar = load_vae(batch, n_samples=n_samples, latent_only = not reconstruction_AD)  

        if reconstruction_AD:
            
            mean_reconstruction_error = compute_loss_continous(reconstructed, batch,None,None,None,AD = True)
            reconstruction_probabilties = compute_probability_continous(reconstructed,batch)

        batch_data = batch.numpy()  
        for i in range(len(batch_data)):
            if reconstruction_AD: 

                if debug < 200:

                    print("recon error from val" , mean_reconstruction_error[i].numpy().item())
                    debug += 1

                reconstruction_errors_threshold.append(mean_reconstruction_error[i].numpy().item())
                reconstruction_probs_threshold.append(reconstruction_probabilties[i].numpy().item())

            if latent_AD:

                distance = bhattacharyya_distance_old(mean_train, variance_train, mu[i], np.exp(logvar[i]))

                #combined_anomaly = np.hstack([mu[i], np.exp(logvar[i])])
                #distances_tree, _ = tree.query([combined_anomaly], k=1)  # Find nearest neighbor
                #distance = distances_tree[0][0]
                normal_distances_threshold.append(distance)
                

    reconsutrction_normal_threshold = 0 
    reconsutrction_probability_threshold = 0            
    latent_normal_threshold = 0
    if latent_AD:
        # Set anomaly threshold 
        #latent_normal_threshold = np.percentile(normal_distances_threshold, 99.5)
        #latent_normal_threshold = np.max(normal_distances_threshold) + np.percentile(normal_distances_threshold, (1 - 0.5)) #<-- maybe better
        latent_normal_threshold = np.mean(normal_distances_threshold) + np.percentile(normal_distances_threshold, (0.05))
        print("########### Latent Thhresholds ##############")

        print(f"Normal LATENT threshold: {latent_normal_threshold:.7f}")
        print("########################################")

    if reconstruction_AD:
        reconsutrction_normal_threshold = np.mean(reconstruction_errors_threshold)
        #reconsutrction_normal_threshold = np.percentile(reconstruction_errors_threshold, 99.5)
        #reconsutrction_normal_threshold = np.max(reconstruction_errors_threshold) * 1.001
        print("########## Reconstruction Thhresholds ##########")
        print(f"Normal Reconstruction threshold: {reconsutrction_normal_threshold:.7f}")

        #reconsutrction_probability_threshold = np.percentile(reconstruction_probs_threshold, 99.5)
        reconsutrction_probability_threshold = np.mean(reconstruction_probs_threshold)

        print(f"Normal Recon Probability threshold: {reconsutrction_probability_threshold:.7f}")
        print("########################################")
    

    print(f"Get Thresold from Train completed in {time.time() - start_time:.4f} seconds")
    return reconsutrction_normal_threshold,reconsutrction_probability_threshold, latent_normal_threshold, mean_train, variance_train, load_vae, tree

def get_mean_variances(model_path, dataset, test = True):   
    start_time = time.time() 
    load_vae = keras.models.load_model(model_path)
    load_vae.trainable = False  # Freeze model weights

    n_samples = 1  # Number of latent samples during inference

    mean_train = []
    variance_train = []
    labels = []

    if test: 
        for batch, label in dataset:
            _, mu, logvar = load_vae(batch, n_samples=n_samples, latent_only = True)  
            for i in range(len(batch)):
                mean_train.append(mu[i])
                variance_train.append(logvar[i])
                labels.append(label[i])
        print(f"Get Mean and Variances completed in: {time.time() - start_time:.4f} seconds")
        return mean_train, variance_train, labels
    else:
        for batch in dataset:
            _, mu, logvar = load_vae(batch, n_samples=n_samples, latent_only = True)  
            for i in range(len(batch)):
                mean_train.append(mu[i])
                variance_train.append(logvar[i])
        print(f"Get Mean and Variances completed in: {time.time() - start_time:.4f} seconds")
        return mean_train, variance_train


"""
The following functions handles distance based AD
"""

def anomaly_detection(load_vae,test_dataset, reconstruction_AD, latent_AD, mean_train, variance_train, tree = None, debug = False):
    start_time = time.time()
    #load_vae = keras.models.load_model(model_path)
    #load_vae.trainable = False  # Freeze model weights

    n_samples = 10  # Number of latent samples during inference

    results = []
    results_probs = []

    distances = []

    debug_count = 0
    for batch, label in test_dataset:
        #print(batch.shape)
        reconstructed, mu, logvar = load_vae(batch, n_samples=n_samples, latent_only = not reconstruction_AD)  # Use multiple samples
        
        # Compute reconstruction errors (mean over all features)
        if reconstruction_AD:
            
            mean_reconstruction_error = compute_loss_continous(reconstructed,batch,None,None,None,AD = True)
            reconstruction_probabilties = compute_probability_continous(reconstructed,batch)

        batch_data = batch.numpy()  # Convert Tensor to NumPy
        for i in range(len(batch_data)):

            if reconstruction_AD:

                results.append(np.append(label[i], mean_reconstruction_error[i].numpy().item()))  # Store the label and max error per sample
                results_probs.append(np.append(label[i], reconstruction_probabilties[i].numpy().item())) 

                if label[i] == 0 and debug < 1000:

                    print("Normal reconstruction: ", mean_reconstruction_error[i].numpy().item())
                    print("Normal Probability: ", reconstruction_probabilties[i].numpy().item())
                    debug +=1  

                if label[i] == 1 and debug < 2000:

                    print("Attack reconstruction: ", mean_reconstruction_error[i].numpy().item())
                    print("Attack Probability: ", reconstruction_probabilties[i].numpy().item())
                    debug +=1 

            if latent_AD:

                distance = bhattacharyya_distance_old(mean_train, variance_train, mu[i], np.exp(logvar[i]))

                #combined_anomaly = np.hstack([mu[i], np.exp(logvar[i])])
                #distances_tree, _ = tree.query([combined_anomaly], k=1)  # Find nearest neighbor
                #distance = distances_tree[0][0]
                distances.append(np.append(label[i], distance))

                if label[i] == 0 and debug < 200:

                    #distance_manual = bhattacharyya_distance_old(mean_train, variance_train, mu[i], np.exp(logvar[i]))
                    #print("Manual Normal Latent ", distance_manual)
                    print("Normal Latent ", distance)
                    debug +=1 

                if label[i] == 1 and debug < 400:

                    #distance_manual = bhattacharyya_distance_old(mean_train, variance_train, mu[i], np.exp(logvar[i]))
                    #print("Manual Attack Latent ", distance_manual)
                    print("Attack latent: ", distance)
                    debug +=1

    print(f"Anomaly Detection completed in : {time.time() - start_time:.4f} seconds")
    return results, results_probs, distances
 

def get_anomaly_detection_accuracy(reconstruction_AD, latent_AD, results, results_probs, reconstruction_normal_threshold, reconstruction_probability_threshold, distances,latent_normal_threshold, 
                                   model_name, latent_dim, epochs, time, n_rows_train, AWS = False, s3 = None, BUCKET = ""):  
    reconstruction_error_accuracy = 0
    reconstruction_probs_accuracy = 0
    latent_accuracy = 0

    if reconstruction_AD:

        copy_results_errors = results.copy()
        copy_results_probs = results_probs.copy()

        # Append anomaly label (1 = anomaly, 0 = normal) directly to `results`
        for i in range(len(results)):

            reconstruction_error = results[i][-1] 
            anomaly_label = 1 if reconstruction_error > reconstruction_normal_threshold else 0  
            copy_results_errors[i] = np.append(results[i], anomaly_label)   

            reconstruction_prob = results_probs[i][-1] 
            anomaly_label = 1 if reconstruction_prob > reconstruction_probability_threshold else 0  
            copy_results_probs[i] = np.append(results_probs[i], anomaly_label)  

        # Print summary
        print("######### Anomalies & Threshold Summary ############")
        print("------ Reconstruction Error -------")
        print(f"Anomaly reconstruction ERROR threshold: {reconstruction_normal_threshold: .4f}")
        print(f"Number of anomalies detected using ERROR: {sum(r[-1] for r in copy_results_errors)}")
        print("------ ------------------- -------")

        print("------ Reconstruction Probability -------")

        print(f"Anomaly reconstruction PROBABILITY threshold: {reconstruction_probability_threshold: .4f}")
        print(f"Number of anomalies detected using PROBABILITY: {sum(r[-1] for r in copy_results_probs)}")
        print("------ ------------------- -------")

        print("######### ############################ ############")

        # Convert results to DataFrame
        columns = []
        columns.append("True_Label")
        columns.append("Reconstruction_Error")  # Add a new column for error
        columns.append("Anomaly")  # Add anomaly label column
        results_df_errors = pd.DataFrame(copy_results_errors, columns=columns)

        # Convert results to DataFrame
        columns = []
        columns.append("True_Label")
        columns.append("Reconstruction_Probability")  # Add a new column for error
        columns.append("Anomaly")  # Add anomaly label column
        results_df_probs = pd.DataFrame(copy_results_probs, columns=columns)

        predicted_errors = results_df_errors['Anomaly'].astype(int)
        true_labels_errors = results_df_errors['True_Label'].astype(int)

        predicted_probs = results_df_probs['Anomaly'].astype(int)
        true_labels_probs = results_df_probs['True_Label'].astype(int)

        excel_file_path = f'./Resources/model_results2.xlsx'

        conf_matrix_errors = confusion_matrix(true_labels_errors, predicted_errors)
        conf_matrix_probs = confusion_matrix(true_labels_probs, predicted_probs)


        # Print results to console
        print("######### Reconstruction Error Performance ############")

        print("Confusion Matrix:")
        print(conf_matrix_errors)

        print("\nPerformance Report:")
        print(classification_report(true_labels_errors, predicted_errors, zero_division= 0))  # Fixed variable name

        #save_results_to_excel(model_name, true_labels, predicted, excel_file_path)
        get_confusion_matrix(true_labels_errors, predicted_errors, latent_dim, epochs, time, n_rows_train, AWS, s3, BUCKET)
        print("######### ############################ ############")

        # Print results to console
        print("######### Reconstruction Probability Performance ############")

        print("Confusion Matrix:")
        print(conf_matrix_probs)

        print("\nPerformance Report:")
        print(classification_report(true_labels_probs, predicted_probs, zero_division= 0))  # Fixed variable name

        #save_results_to_excel(model_name, true_labels, predicted, excel_file_path)
        get_confusion_matrix(true_labels_probs, predicted_probs, latent_dim, epochs, time, n_rows_train, AWS, s3, BUCKET)
        print("######### ############################ ############")


        reconstruction_error_accuracy = np.mean(predicted_errors == true_labels_errors)
        reconstruction_probs_accuracy = np.mean(predicted_probs == true_labels_probs)

    if latent_AD:
        copy_distances = distances.copy()
        # Look if distances is over normal threshold
        for i in range(len(distances)):
            latent_distance = distances[i][-1] 
            anomaly_label = 1 if latent_distance > 0.5 else 0  
            copy_distances[i] = np.append(distances[i], anomaly_label)  

        # Print summary
        print(f"Anomaly threshold: {latent_normal_threshold:.7f}")
        print(f"Number of anomalies detected using LATENT: {sum(r[-1] for r in copy_distances)}")

        # Convert results to DataFrame
        columns = []
        columns.append("True_Label")
        columns.append("Distance")  # Add a new column for error
        columns.append("Anomaly")  # Add anomaly label column
        distances_df = pd.DataFrame(copy_distances, columns=columns)

        predicted_latent = distances_df['Anomaly'].astype(int)
        true_labels_latent = distances_df['True_Label'].astype(int)

        excel_file_path = f'./Resources/model_results2.xlsx'
        conf_matrix_latent = confusion_matrix(true_labels_latent, predicted_latent)

        print("######### Latent Distance Performance ############")
        print("Confusion Matrix:")
        print(conf_matrix_latent)


        #save_results_to_excel(model_name, true_labels, predicted, excel_file_path)
        get_confusion_matrix(true_labels_latent, predicted_latent, latent_dim, epochs, time, n_rows_train, AWS, s3, BUCKET)
        print("######### ############################ ############")

        latent_accuracy = np.mean(predicted_latent == true_labels_latent)
    
    return reconstruction_error_accuracy , reconstruction_probs_accuracy, latent_accuracy



"""
The following functions handles distance calculations
"""

def bhattacharyya_distance(mu_normals, logvar_normals, mu_anomaly, logvar_anomaly):
    # Ensure mu_normals, mu_anomaly, logvar_normals, logvar_anomaly are NumPy arrays
    #mu_normals = mu_normals.numpy() if isinstance(mu_normals, tf.Tensor) else mu_normals
    #mu_anomaly = mu_anomaly.numpy() if isinstance(mu_anomaly, tf.Tensor) else mu_anomaly
    #logvar_normals = logvar_normals.numpy() if isinstance(logvar_normals, tf.Tensor) else logvar_normals
    #logvar_anomaly = logvar_anomaly.numpy() if isinstance(logvar_anomaly, tf.Tensor) else logvar_anomaly

    #sigma2_normals = np.exp(logvar_normals)  # normal variance
    sigma2_normals = logvar_normals
    sigma2_anomaly = np.exp(logvar_anomaly)  # anomaly variance

    # Check for very small variances (which might cause instability in log or sqrt)
    #if np.any(sigma2_normals < 1e-10) or np.any(sigma2_anomaly < 1e-10):
    #    print("Warning: Small variances detected!")
    #    print("sigma2_normals:", sigma2_normals)
    #    print("sigma2_anomaly:", sigma2_anomaly)

    diff_sq = (mu_normals - mu_anomaly) ** 2

    #if np.any(diff_sq < 0):
    #    print("diff_sq negative!!:")


    # Compute term1
    term1 = 0.25 * np.sum(diff_sq / (sigma2_normals + sigma2_anomaly), axis=1)
    #if np.any(term1 < 0):
    #    print("term 1 negative!!:")
    eps = 1e-10  # Small epsilon to avoid taking log of 0
    # Compute term2
    # Ensure no division by zero and sqrt of negative values
    #term2 = 0.5 * np.sum(np.log(((sigma2_normals + sigma2_anomaly) / 2) / np.sqrt(sigma2_normals * sigma2_anomaly)), axis=1)
    #term2 = 0.5 * np.sum(np.log(((sigma2_normals + sigma2_anomaly) / 2) / 
    #                            np.sqrt(np.maximum(sigma2_normals, eps) * np.maximum(sigma2_anomaly, eps))), axis=1)
    # Avoid log(0) and sqrt of negative values by adding epsilon for more stability
    eps = np.finfo(np.float32).eps
    term2 = 0.5 * np.sum(np.log(((sigma2_normals + sigma2_anomaly) / 2) / 
                                np.sqrt(np.maximum(sigma2_normals, eps) * np.maximum(sigma2_anomaly, eps))), axis=1)
    #if np.any(term2 < 0):
        #print("term 2 negative!!:")



    # Compute final distances
    distances = term1 + term2
    return np.min(distances)

def bhattacharyya_distance_old(mu_normals, logvar_normals, mu_anomaly, logvar_anomaly):
    # Function compares how much distributions overlap
    
    #print("manual mu len", mu_anomaly.shape)
    #print("manual log len", logvar_anomaly.shape)
    #print("manual mu len", mu_anomaly.shape)
    #print("manual log len", logvar_anomaly.shape)
    
    #sigma2_normals = np.exp(logvar_normals)  # normal variance
    sigma2_normals = logvar_normals
    sigma2_anomaly = logvar_anomaly  # anomaly variance

    diff_sq = (mu_normals - mu_anomaly) ** 2
    term1 = 0.25 * np.sum(diff_sq / (sigma2_normals + sigma2_anomaly), axis=1)

    term2 = 0.5 * np.sum(np.log(((sigma2_normals + sigma2_anomaly) / 2) / np.sqrt(sigma2_normals * sigma2_anomaly)), axis=1)

    distances = term1 + term2
    distance = np.min(distances[distances > 0])
    return  distance

def bhattacharyya_distance_old_balltree(x, y):
    # Function compares how much distributions overlap
    latent_dim = len(x) // 2  # Determine the latent dimension
    mu_normals, logvar_normals = x[:latent_dim], x[latent_dim:]
    mu_anomaly, logvar_anomaly = y[:latent_dim], y[latent_dim:]

    sigma2_normals = logvar_normals
    sigma2_anomaly = logvar_anomaly
    
    #print("BALL Tree mu len", mu_anomaly.shape)
    #print("BALL Tree log len", logvar_anomaly.shape)
    #print("-----")
    #print("BALL Tree mu len", mu_anomaly.shape)
    #print("BALL Tree log len", logvar_anomaly.shape)
    #print("----")
    #sigma2_normals = np.exp(logvar_normals)  # normal variance
    #sigma2_anomaly = np.exp(logvar_anomaly)  # anomaly variance

    diff_sq = (mu_normals - mu_anomaly) ** 2
    term1 = 0.25 * np.sum(diff_sq / (sigma2_normals + sigma2_anomaly))

    term2 = 0.5 * np.sum(np.log(((sigma2_normals + sigma2_anomaly) / 2) / np.sqrt(sigma2_normals * sigma2_anomaly)))

    distances = term1 + term2
    return distances


def gaussian_distance(mean1, var1, mean2, var2):
    """
    Calculate Wasserstein-2 distance between Gaussians
    (Simplified for diagonal covariance matrices)
    """
    # Mean term
    mean_diff = np.sum((mean1 - mean2)**2)
    
    # Variance term (for diagonal covariance)
    var_diff = np.sum((np.sqrt(var1) - np.sqrt(var2))**2)
    
    return np.sqrt(mean_diff + var_diff)


def mahalanobis_distance(mu_normals, logvar_normals, mu_anomaly, logvar_anomaly):
    # Convert log-variance to actual variance
    #var_normals = np.exp(logvar_normals)
    var_normals = logvar_normals
    # Compute the difference between anomaly and normal mean
    delta = mu_anomaly - mu_normals
    
    # Mahalanobis distance with diagonal covariance
    mahalanobis = np.sqrt(np.sum((delta**2) / var_normals, axis=-1))
    distance = np.min(mahalanobis[mahalanobis > 0])
    return distance


def kl_divergence_gaussians(mu_normals, logvar_normals, mu_anomaly, logvar_anomaly):
    # Compute directly with log-domain arithmetic where possible
    term1 = np.sum(np.exp(logvar_normals - logvar_anomaly), axis=1)  # More stable variance ratio
    term2 = np.sum((mu_anomaly - mu_normals) ** 2 / np.exp(logvar_anomaly), axis=1)  # Mean difference squared
    term3 = np.sum(logvar_anomaly - logvar_normals, axis=1)  # Log variance ratio
    kl_div = 0.5 * (term1 + term2 + term3)
    distance = np.min(kl_div[kl_div > 0])
    return distance
