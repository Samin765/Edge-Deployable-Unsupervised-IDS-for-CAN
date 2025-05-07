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

def get_mean_variances(dataset, test = False, load_vae = None, model_path = ""):   
    start_time = time.time() 

    if load_vae == None: 
        load_vae = keras.models.load_model(model_path)
        load_vae.trainable = False  # Freeze model weights

    n_samples = 10  # Number of latent samples during inference

    mean_train = []
    variance_train = []
    labels = []

    if test: 
        for batch, label in dataset:
            model_outputs = load_vae(batch, n_samples=n_samples, latent_only = True)  

            mu = model_outputs['mu']
            logvar = model_outputs['logvar']
            
            for i in range(len(batch)):
                mean_train.append(mu[i])
                variance_train.append(np.exp(logvar[i]))
                labels.append(label[i])

        print(f"Get Mean and Variances completed in: {time.time() - start_time:.4f} seconds")
        return mean_train, variance_train, labels
    else:
        for batch in dataset:
            model_outputs = load_vae(batch, n_samples=n_samples, latent_only = True)

            mu = model_outputs['mu']
            logvar = model_outputs['logvar']  

            for i in range(len(batch)):
                mean_train.append(mu[i])
                variance_train.append(np.exp(logvar[i]))
        print(f"Get Mean and Variances completed in: {time.time() - start_time:.4f} seconds")
        return mean_train, variance_train

def get_threshold_from_train(model_path, train_dataset, val_dataset, reconstruction_AD, latent_AD, binary = False, val_dataset2 = None):    
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
        """

        for batch in train_dataset:

            reconstructed, mu, logvar = load_vae(batch, n_samples=n_samples, latent_only = latent_AD)  
            batch_data = batch.numpy() 

            for i in range(len(batch)):

                mean_train.append(mu[i])
                variance_train.append(np.exp(logvar[i]))

        print(f"Collect Mean & Variance from Train completed in {time.time() - start_time:.4f} seconds")
        """
        mean_train , variance_train = get_mean_variances(train_dataset, test = False, load_vae = load_vae, model_path= "")

        # Preprocess your normal data
        start_time_ball_tree = time.time()
        combined_params = np.hstack([mean_train, variance_train])  # Combine mean and variance
        tree = BallTree(combined_params, metric='pyfunc', func=bhattacharyya_distance_old_balltree)
        print(f"Ball Tree completed in {time.time() - start_time_ball_tree:.4f} seconds")

    debug = 0
    for batch in val_dataset:

        model_outputs = load_vae(batch, n_samples=n_samples, latent_only = not reconstruction_AD)  

        reconstructed = model_outputs['reconstructed']
        mu = model_outputs['mu']
        logvar = model_outputs['logvar']

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

                #distance = bhattacharyya_distance_old(mean_train, variance_train, mu[i], np.exp(logvar[i]))
                distance = min_euclidean_distance(mean_train, mu[i])
                if debug < 200:

                    print("Distance from val" , distance)
                    debug += 1
                #combined_anomaly = np.hstack([mu[i], np.exp(logvar[i])])
                #distances_tree, _ = tree.query([combined_anomaly], k=1)  # Find nearest neighbor
                #distance = distances_tree[0][0]
                normal_distances_threshold.append(distance)
                

    reconsutrction_normal_threshold = 0 
    reconsutrction_probability_threshold = 0            
    latent_normal_threshold = 0
    if latent_AD:
        # Set anomaly threshold s
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

def get_threshold_from_test(model_path, train_dataset, val_dataset, reconstruction_AD, latent_AD, binary = False, val_dataset2 = None):    
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
        """

        for batch in train_dataset:

            reconstructed, mu, logvar = load_vae(batch, n_samples=n_samples, latent_only = latent_AD)  
            batch_data = batch.numpy() 

            for i in range(len(batch)):

                mean_train.append(mu[i])
                variance_train.append(np.exp(logvar[i]))

        print(f"Collect Mean & Variance from Train completed in {time.time() - start_time:.4f} seconds")
        """
        mean_train , variance_train , labels = get_mean_variances(train_dataset, test = True, load_vae = load_vae, model_path= "")

        # Preprocess your normal data
        start_time_ball_tree = time.time()
        combined_params = np.hstack([mean_train, variance_train])  # Combine mean and variance
        tree = BallTree(combined_params, metric='pyfunc', func=bhattacharyya_distance_old_balltree)
        print(f"Ball Tree completed in {time.time() - start_time_ball_tree:.4f} seconds")

    debug = 0
    for batch, labels in val_dataset:

        model_outputs = load_vae(batch, labels,n_samples=n_samples, latent_only = not reconstruction_AD)  

        reconstructed = model_outputs['reconstructed']
        mu = model_outputs['mu']
        logvar = model_outputs['logvar']
        hidden = model_outputs['hidden']
        y_pred = model_outputs['y_pred']

        if reconstruction_AD:
            
            recon_loss_batch = compute_loss_continous(reconstructed, batch,None,None,None,AD = True)
            losses = load_vae.compute_loss(labels, recon_loss_batch , hidden , y_pred, AD = True)

            reconstruction_probabilties = compute_probability_continous(reconstructed,batch)
            classification_loss = losses['classification_loss']
            masked_recon_loss_batch = losses['masked_recon_loss']
            kappa = losses['kappa']

            mean_reconstruction_error = masked_recon_loss_batch

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
                if debug < 200:

                    print("Distance from val" , distance)
                    debug += 1
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
        model_outputs = load_vae(batch, n_samples=n_samples, latent_only = not reconstruction_AD)  # Use multiple samples

        reconstructed = model_outputs['reconstructed']
        mu = model_outputs['mu']
        logvar = model_outputs['logvar']
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

                #distance = bhattacharyya_distance_old(mean_train, variance_train, mu[i], np.exp(logvar[i]))
                distance = min_euclidean_distance(mean_train, mu[i])

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
            anomaly_label = 1 if reconstruction_prob > 37.5 else 0  
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
            anomaly_label = 1 if latent_distance > latent_normal_threshold else 0  
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
    sigma2_anomaly = logvar_anomaly  # anomaly variance

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
    #distance = np.min(distances)
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


def min_euclidean_distance(mean_list, test_mean):
    """
    Compute the minimum Euclidean distance between a test mean vector
    and a list/array of mean vectors.
    
    Parameters:
    - mean_list: np.ndarray of shape (N, D), where N is the number of mean vectors
    - test_mean: np.ndarray of shape (D,)
    
    Returns:
    - min_distance: float, the smallest Euclidean distance found
    """
    mean_list = np.asarray(mean_list)
    test_mean = np.asarray(test_mean)
    
    # Compute Euclidean distances
    distances = np.linalg.norm(mean_list - test_mean, axis=1)
    
    # Return the minimum distance
    return np.min(distances[distances > 0])

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


def euclidean_distance(mean1, mean2):
    """
    Calculate Euclidean distance between two mean vectors.
    """
    return np.linalg.norm(mean1 - mean2)

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
