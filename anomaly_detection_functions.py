import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix, classification_report
from utils import get_confusion_matrix

def get_threshold_from_train(model_path, train_dataset, val_dataset,reconstruction_AD, latent_AD):    
    load_vae = keras.models.load_model(model_path)
    load_vae.trainable = False  # Freeze model weights

    # Compute reconstruction losses on the test set with multiple samples
    n_samples = 1  # Number of latent samples during inference
    reconstruction_errors_threshold = []

    mean_train = []
    variance_train = []

    normal_distances_threshold = []
    if latent_AD:
        for batch in train_dataset:
            reconstructed, mu, logvar = load_vae(batch, n_samples=n_samples, latent_only = latent_AD)  
            batch_data = batch.numpy()  
            for i in range(len(batch)):
                mean_train.append(mu[i])
                variance_train.append(logvar[i])

                
    for batch in val_dataset:
        reconstructed, mu, logvar = load_vae(batch, n_samples=n_samples, latent_only = not reconstruction_AD)  

        if reconstruction_AD:
            # Compute VAE loss
            reconstruction_errors = tf.reduce_mean(
                tf.square(tf.expand_dims(batch, axis=0) - reconstructed), axis = -1
            )  # Shape: (n_samples, batch_size, window_size)  

            # mean over samples and window_size
            mean_reconstruction_error = tf.reduce_mean(reconstruction_errors, axis=(2, 0))
            #max_reconstruction_error = tf.reduce_max(reconstruction_errors, axis=(2, 0))

        batch_data = batch.numpy()  
        for i in range(len(batch)):
            if reconstruction_AD: 
                reconstruction_errors_threshold.append(mean_reconstruction_error[i])
            if latent_AD:
                distance = bhattacharyya_distance(mean_train, variance_train, mu[i], logvar[i])
                normal_distances_threshold.append(distance)

    reconsutrction_normal_threshold = 0             
    latent_normal_threshold = 0
    if latent_AD:
        # Set anomaly threshold 
        #latent_normal_threshold = np.percentile(normal_distances_threshold, 99.5)
        #latent_normal_threshold = np.max(normal_distances_threshold) + np.percentile(normal_distances_threshold, (1 - 0.5)) #<-- maybe better
        latent_normal_threshold = np.max(normal_distances_threshold) * 1.001
        print(f"Normal LATENT threshold: {latent_normal_threshold:.7f}")

    if reconstruction_AD:
        #reconsutrction_normal_threshold = np.mean(reconstruction_errors_threshold) + np.percentile(reconstruction_errors_threshold, (1 - 0.5))
        reconsutrction_normal_threshold = np.percentile(reconstruction_errors_threshold, 99.5)
        print(f"Normal Reconstruction threshold: {reconsutrction_normal_threshold:.7f}")
    
    return reconsutrction_normal_threshold, latent_normal_threshold, mean_train, variance_train


def anomaly_detection(load_vae,test_dataset, reconstruction_AD, latent_AD, mean_train, variance_train, debug = False):
    #load_vae = keras.models.load_model(model_path)
    #load_vae.trainable = False  # Freeze model weights

    n_samples = 1  # Number of latent samples during inference

    results = []
    reconstruction_losses = []

    distances = []

    debug_count = 0
    for batch, label in test_dataset:
        #print(batch.shape)
        reconstructed, mu, logvar = load_vae(batch, n_samples=n_samples, latent_only = not reconstruction_AD)  # Use multiple samples
        
        # Compute reconstruction errors (mean over all features)
        if reconstruction_AD:
            #print(batch.shape)
            errors = tf.reduce_mean(tf.square(tf.expand_dims(batch, axis=0) - reconstructed), axis=-1)  # Shape: (n_samples, batch_size)

            # Compute mean reconstruction error across samples (axis=0) and features (axis=-1)
            mean_reconstruction_error = tf.reduce_mean(errors, axis=(0, 2))  # Shape: (batch_size,)
            #mean_reconstruction_error = tf.reduce_max(errors, axis=(0, 2))  # Shape: (batch_size,)
            #print(mean_reconstruction_error.shape)
            reconstruction_losses.extend(mean_reconstruction_error)

        batch_data = batch.numpy()  # Convert Tensor to NumPy
        for i in range(len(batch_data)):
            if reconstruction_AD:
                results.append(np.append(label[i], reconstruction_losses[i].numpy().item()))  # Store the label and max error per sample
                #if label[i] == 0 and debug < 10:
                #    print("Normal reconstruction: ", reconstruction_losses[i].numpy().item())
                #    debug +=1  
                #if label[i] == 1 and debug < 200:
                #    print("Attack reconstruction: ", reconstruction_losses[i].numpy().item())
                #    debug +=1 
            if latent_AD:
                distance = bhattacharyya_distance(mean_train, variance_train, mu[i], logvar[i])
                #print("mu" , len(mu))
                #print("mu" , len(logvar))
                distances.append(np.append(label[i], distance))
                #if label[i] == 0 and debug < 20:
                #    print("Normal Latent ", distance)
                #    debug +=1 
                #if label[i] == 1 and debug < 40:
                #    print("Attack latent: ", distance)
                #    debug +=1
    return results, distances
 

def get_anomaly_detection_accuracy(reconstruction_AD, latent_AD, results, reconsutrction_normal_threshold, distances,latent_normal_threshold, 
                                   model_name, latent_dim, epochs, time, n_rows_train, AWS = False, s3 = None, BUCKET = ""):  
    reconstruction_accuracy = 0
    latent_accuracy = 0
    if reconstruction_AD:
        copy_results = results.copy()
        # Append anomaly label (1 = anomaly, 0 = normal) directly to `results`
        for i in range(len(results)):
            reconstruction_error = results[i][-1] 
            anomaly_label = 1 if reconstruction_error < reconsutrction_normal_threshold else 0  
            copy_results[i] = np.append(results[i], anomaly_label)  

        # Print summary
        print(f"Anomaly RECONSTRUCTION threshold: {reconsutrction_normal_threshold:.4f}")
        print(f"Number of anomalies detected using RECONSTRUCTED: {sum(r[-1] for r in copy_results)}")

        # Convert results to DataFrame
        columns = []
        columns.append("True_Label")
        columns.append("Reconstruction_Error")  # Add a new column for error
        columns.append("Anomaly")  # Add anomaly label column
        results_df = pd.DataFrame(copy_results, columns=columns)

        predicted = results_df['Anomaly'].astype(int)
        true_labels = results_df['True_Label'].astype(int)

        excel_file_path = f'/Users/SCHUGD/Desktop/MasterThesis/Code/model_results2.xlsx'
        conf_matrix = confusion_matrix(true_labels, predicted)

        # Print results to console
        print("Confusion Matrix:")
        print(conf_matrix)

        print("\nPerformance Report:")
        print(classification_report(true_labels, predicted, zero_division= 0))  # Fixed variable name

        #save_results_to_excel(model_name, true_labels, predicted, excel_file_path)
        get_confusion_matrix(true_labels, predicted, latent_dim, epochs, time, n_rows_train, AWS, s3, BUCKET)

        reconstruction_accuracy = np.mean(predicted == true_labels)

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

        predicted = distances_df['Anomaly'].astype(int)
        true_labels = distances_df['True_Label'].astype(int)

        excel_file_path = f'/Users/SCHUGD/Desktop/MasterThesis/Code/model_results2.xlsx'

        #save_results_to_excel(model_name, true_labels, predicted, excel_file_path)
        get_confusion_matrix(true_labels, predicted, latent_dim, epochs, time, n_rows_train, AWS, s3, BUCKET)
        
        latent_accuracy = np.mean(predicted == true_labels)
    
    return reconstruction_accuracy , latent_accuracy


def bhattacharyya_distance(mu_normals, logvar_normals, mu_anomaly, logvar_anomaly):
    # Ensure mu_normals, mu_anomaly, logvar_normals, logvar_anomaly are NumPy arrays
    #mu_normals = mu_normals.numpy() if isinstance(mu_normals, tf.Tensor) else mu_normals
    #mu_anomaly = mu_anomaly.numpy() if isinstance(mu_anomaly, tf.Tensor) else mu_anomaly
    #logvar_normals = logvar_normals.numpy() if isinstance(logvar_normals, tf.Tensor) else logvar_normals
    #logvar_anomaly = logvar_anomaly.numpy() if isinstance(logvar_anomaly, tf.Tensor) else logvar_anomaly

    sigma2_normals = np.exp(logvar_normals)  # normal variance
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
    
    sigma2_normals = np.exp(logvar_normals)  # normal variance
    sigma2_anomaly = np.exp(logvar_anomaly)  # anomaly variance

    diff_sq = (mu_normals - mu_anomaly) ** 2
    term1 = 0.25 * np.sum(diff_sq / (sigma2_normals + sigma2_anomaly), axis=1)

    term2 = 0.5 * np.sum(np.log(((sigma2_normals + sigma2_anomaly) / 2) / np.sqrt(sigma2_normals * sigma2_anomaly)), axis=1)

    distances = term1 + term2
    return np.min(distances)  