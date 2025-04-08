from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import hdbscan
from sklearn.metrics import precision_recall_curve, auc, roc_curve, f1_score
from hdbscan.prediction import approximate_predict
from sklearn.manifold import TSNE
import keras


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

