import tensorflow as tf
from utils import get_latent_representations_label, linear_annealing, save_trained_model

"""
Beta-VAE based on:

"β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
(https://arxiv.org/abs/1804.03599)
"""
def train_model(vae,optimizer,discriminator_optimizer, epochs, n_samples, input_dim, latent_dim, 
                batch_size,beta, gamma, n_critic, steps_anneal, patience, time,beta_tc = 0, validation_method = 'None', 
                model_path = "", train_dataset = None,test_dataset = None,val_dataset = None, n_rows_train = 0, AWS = False,
                s3 = None, BUCKET = "" ):   
    wait = 0  
    model_name ="BetaVAE"

    print(f"Latent Dimension = {latent_dim}, "
        f"Beta = {beta}, Validation Method = {validation_method}, "
        f"Rows in Training Data = {n_rows_train}, "
        f"Batch Size = {batch_size}") 
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    real_epochs = 0

    # Training loop
    for epoch in range(epochs):
        real_epochs += 1
        epoch_loss = 0

        for step, batch in enumerate(train_dataset): # window size , features
            global_step = epoch * len(train_dataset) + step  # Total training step count

            anneal_coeff = linear_annealing(0, 1, global_step, steps_anneal)

            # Train VAE
            with tf.GradientTape() as tape:
                reconstructed, mu, logvar = vae(batch, n_samples=n_samples, latent_only = False)  # Use multiple samples
                
                expanded_batch = tf.expand_dims(batch, axis = 0)
                binary_features = 29

                # Extract features
                batch_binary = expanded_batch[..., :binary_features]
                reconstructed_binary = reconstructed[..., :binary_features]
                batch_continuous = expanded_batch[..., binary_features:]
                reconstructed_continuous = reconstructed[..., binary_features:]

                # BCE loss for binary features
                bce_errors = tf.keras.losses.BinaryCrossentropy(reduction='none')(
                    batch_binary, reconstructed_binary)

                #Compute reconstruction MSE error for continous features
                mse_errors = tf.reduce_mean(
                    tf.square(batch_continuous - reconstructed_continuous), axis=-1
                )  # Shape: (n_samples, batch_size, window_size)

                reconstruction_errors = bce_errors + mse_errors

                # Aggregate errors (mean over samples and window_size)
                mean_reconstruction_error = tf.reduce_mean(reconstruction_errors, axis=(2, 0))

                # Compute VAE loss
                reconstruction_loss = tf.reduce_mean(mean_reconstruction_error)

                #print(f"Final loss {reconstruction_loss}")
                kl_divergence = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
                  
                loss = reconstruction_loss + beta * kl_divergence

            gradients = tape.gradient(loss, vae.encoder.trainable_variables + vae.decoder.trainable_variables 
                                      + vae.fc_mu.trainable_variables + vae.fc_logvar.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimizer.apply_gradients(zip(gradients, vae.encoder.trainable_variables + vae.decoder.trainable_variables 
                                          + vae.fc_mu.trainable_variables + vae.fc_logvar.trainable_variables))

            epoch_loss += loss.numpy()

        # VALIDATION
        epoch_val_loss = 0
        show_val = validation_method in ["B_VAE"]

        if validation_method == "None" or validation_method == "PLOT":
            val_loss = 0
            early_stop = False
        else:
            early_stop = True
            for batch in val_dataset:
                reconstructed, mu, logvar = vae(batch, n_samples=n_samples, latent_only=False)

                reconstruction_errors = tf.reduce_mean(
                    tf.square(tf.expand_dims(batch, axis=0) - reconstructed), axis=-1
                )
                mean_reconstruction_error = tf.reduce_mean(reconstruction_errors, axis=(2, 0))
                reconstruction_loss = tf.reduce_mean(mean_reconstruction_error)
                kl_divergence = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))

                val_loss = reconstruction_loss + beta * kl_divergence
                epoch_val_loss += val_loss.numpy()


        if validation_method == "PLOT" and epoch % 50 == 0 and epoch > 0:
            print("PLOT AT EPOCH: ", {epoch + 1})
            get_latent_representations_label(vae, test_dataset, latent_dim, beta,n_critic,gamma,time,epoch = epoch, name = model_name,type = 'TSNE', save = False, AWS = AWS, s3 = s3, BUCKET = BUCKET)


        # Store the loss for this epoch
        train_loss = epoch_loss / len(train_dataset)
        val_loss = epoch_val_loss / len(val_dataset) 

        train_losses.append(train_loss)
        val_losses.append(val_loss)


        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, "
            f"Val Loss: {val_loss:.6f}")
        
        if early_stop:
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                # Save best model
                save_trained_model(vae, optimizer, model_path, model_name, latent_dim,beta,n_rows_train,time,epoch, AWS = AWS, s3 = s3 , BUCKET = BUCKET)
                print( f'💽Saved Model at Epoch {epoch+1}💽')
            else:
                wait += 1
                if wait >= patience:
                    epochs = epoch + 1
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
        # Save the trained model each 10th epoch
        if epoch % 50 == 0 and epoch > 0:
            save_trained_model(vae, optimizer, model_path, model_name, latent_dim,beta,n_rows_train,time,epoch, AWS = AWS, s3 = s3 , BUCKET = BUCKET)
            print( f'💽Saved Model at Epoch {epoch+1}💽')

    save_trained_model(vae, optimizer, model_path, model_name, latent_dim,beta,n_rows_train,time,epoch, AWS = AWS, s3 = s3 , BUCKET = BUCKET)
    print( f'💽Saved Model at Epoch {epoch+1}💽')

    print("💽Loss Plot Saved.💽")
    print("💽VAE model saved.💽")
    #vae.summary()
    return train_losses, val_losses, real_epochs, time, show_val, model_path, vae



"""
Factor-VAE based on:

"Disentangling by Factorising"
(https://arxiv.org/abs/1802.05983)
"""
def train_model_factor(vae,optimizer,discriminator_optimizer, epochs, n_samples, input_dim, latent_dim, 
                batch_size,beta, gamma, n_critic, steps_anneal, patience, time,beta_tc = 0, validation_method = 'None', 
                model_path = "", train_dataset = None,test_dataset = None,val_dataset = None, n_rows_train = 0, AWS = False,
                s3 = None, BUCKET = "" ):   
    wait = 0  
    model_name ="FactorVAE"

    print(f"Latent Dimension = {latent_dim}, "
        f"Beta = {beta}, Gamma = {gamma}, N_critic = {n_critic}, Validation Method = {validation_method}, "
        f"Rows in Training Data = {n_rows_train}, "
        f"Batch Size = {batch_size}") 
    train_losses = []
    val_losses = []
    disc_losses = []
    best_val_loss = float('inf')
    real_epochs = 0

    # Training loop
    for epoch in range(epochs):
        real_epochs += 1
        epoch_loss = 0
        epoch_disc_loss = 0
        for step, batch in enumerate(train_dataset): # window size , features
            global_step = epoch * len(train_dataset) + step  # Total training step count

            anneal_coeff = linear_annealing(0, 1, global_step, steps_anneal)

            # Split the batch 
            half_batch_size = tf.shape(batch)[0] // 2
            first_half_batch = batch[:half_batch_size]
            second_half_batch = batch[half_batch_size:]

            # 1. Train VAE with first batch
            with tf.GradientTape() as tape:
                reconstructed, mu, logvar = vae(first_half_batch, n_samples=n_samples, latent_only = False)  # Use multiple samples

                #Compute reconstruction MSE error for each sample
                reconstruction_errors = tf.reduce_mean(
                    tf.square(tf.expand_dims(first_half_batch, axis=0) - reconstructed), axis=-1
                                            )  # Shape: (n_samples, batch_size, window_size)

                # Aggregate errors (mean over samples and window_size)
                mean_reconstruction_error = tf.reduce_mean(reconstruction_errors, axis=(2, 0))

                # Compute VAE loss
                reconstruction_loss = tf.reduce_mean(mean_reconstruction_error)

                kl_divergence = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
                                
                # Get latent samples for TC loss
                z = vae.reparameterize(mu, logvar, n_samples=n_samples)
                z = tf.reduce_mean(z, axis=0)  # Remove sample dimension for discriminator
                    
                # Total Correlation loss
                tc = vae.tc_loss(z)
                loss = reconstruction_loss + beta * kl_divergence + gamma*tc*anneal_coeff

            gradients = tape.gradient(loss, vae.encoder.trainable_variables + vae.decoder.trainable_variables 
                                      + vae.fc_mu.trainable_variables + vae.fc_logvar.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimizer.apply_gradients(zip(gradients, vae.encoder.trainable_variables + vae.decoder.trainable_variables 
                                          + vae.fc_mu.trainable_variables + vae.fc_logvar.trainable_variables))

            epoch_loss += loss.numpy()
            
            # 2. Train discriminator with second batch
            disc_loss = 0
            for _ in range(n_critic):
                # Discriminator update
                with tf.GradientTape() as disc_tape:

                    # First get latent representations on second half batch
                    _, mu, logvar = vae(second_half_batch, n_samples=n_samples, latent_only=True)
                    z = vae.reparameterize(mu, logvar, n_samples=n_samples)
                    z = tf.reduce_mean(z, axis=0)
                    z_perm = vae.permute_dims(z)
                    current_disc_loss = vae.discriminator_loss(z, z_perm)
                    disc_loss += current_disc_loss

                # Update discriminator
                disc_gradients = disc_tape.gradient(current_disc_loss, vae.discriminator.trainable_variables)
                # Clip gradients for stability
                disc_gradients, _ = tf.clip_by_global_norm(disc_gradients, 1.0)
                discriminator_optimizer.apply_gradients(zip(disc_gradients, vae.discriminator.trainable_variables))

            disc_loss = disc_loss / n_critic
            epoch_disc_loss += disc_loss
        
        # VALIDATION
        epoch_val_loss = 0
        total_disc_acc = 0
        show_val = validation_method in ["B_VAE", "TC"]
        latent_only = validation_method == "TC"

        if validation_method == "None" or validation_method == "PLOT":
            val_loss = 0
            avg_disc_acc = 0
            early_stop = False
        else:
            early_stop = True
            for batch in val_dataset:
                reconstructed, mu, logvar = vae(batch, n_samples=n_samples, latent_only=latent_only)

                if validation_method == "B_VAE":
                    reconstruction_errors = tf.reduce_mean(
                        tf.square(tf.expand_dims(batch, axis=0) - reconstructed), axis=-1
                    )
                    mean_reconstruction_error = tf.reduce_mean(reconstruction_errors, axis=(2, 0))
                    reconstruction_loss = tf.reduce_mean(mean_reconstruction_error)
                    kl_divergence = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))

                    val_loss = reconstruction_loss + beta * kl_divergence
                    epoch_val_loss += val_loss.numpy()
                elif validation_method == "TC":
                    z = vae.reparameterize(mu, logvar, n_samples=n_samples)
                    z = tf.reduce_mean(z, axis= 0)
                    tc = vae.tc_loss(z)
                    
                    disc_acc = vae.discriminator_acc(z)
                    total_disc_acc += disc_acc
                    val_loss = gamma * tc
                    epoch_val_loss += val_loss.numpy()

        if validation_method == "PLOT" and epoch % 50 == 0 and epoch > 0:
            print("PLOT AT EPOCH: ", {epoch + 1})
            get_latent_representations_label(vae, test_dataset, latent_dim, beta,n_critic,gamma,time,epoch = epoch, name = model_name,type = 'TSNE', save = False, AWS = AWS, s3 = s3, BUCKET = BUCKET)


        # Store the loss for this epoch
        train_loss = epoch_loss / len(train_dataset)
        val_loss = epoch_val_loss / len(val_dataset) 
        disc_loss = epoch_disc_loss / len(train_dataset)
        avg_disc_acc = total_disc_acc / len(val_dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        disc_losses.append(disc_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, "
            f"Discriminator Loss: {disc_loss:.6f} Disc Acc: {avg_disc_acc:.4f}, Val Loss: {val_loss:.6f}")
        
        if early_stop:
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                # Save best model
                save_trained_model(vae, optimizer, model_path, model_name, latent_dim,beta,n_rows_train,time,epoch, AWS = AWS, s3 = s3 , BUCKET = BUCKET)
                print( f'💽Saved Model at Epoch {epoch+1}💽')
            else:
                wait += 1
                if wait >= patience:
                    epochs = epoch + 1
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
        # Save the trained model
        if epoch % 50 == 0 and epoch > 0:
            save_trained_model(vae, optimizer, model_path, model_name, latent_dim,beta,n_rows_train,time,epoch, AWS = AWS, s3 = s3 , BUCKET = BUCKET)
            print( f'💽Saved Model at Epoch {epoch+1}💽')

    save_trained_model(vae, optimizer, model_path, model_name, latent_dim,beta,n_rows_train,time,epoch, AWS = AWS, s3 = s3 , BUCKET = BUCKET)
    print( f'💽Saved Model at Epoch {epoch+1}💽')

    print("💽Loss Plot Saved.💽")
    print("💽VAE model saved.💽")
    #vae.summary()
    return train_losses, val_losses, real_epochs, time, show_val, model_path, vae




"""
β-TCVAE based on:

"Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/abs/1802.04942)
"""
def train_model_btc(vae,optimizer,discriminator_optimizer, epochs, n_samples, input_dim, latent_dim, 
                batch_size,beta, gamma, n_critic, steps_anneal, patience, time,beta_tc = 0, validation_method = 'None', 
                model_path = "", train_dataset = None,test_dataset = None,val_dataset = None, n_rows_train = 0, AWS = False,
                s3 = None, BUCKET = "" ):   
    wait = 0  
    model_name ="BtcVAE"

    print(f"Latent Dimension = {latent_dim}, "
        f"Beta_TC = {beta_tc}, Validation Method = {validation_method}, "
        f"Rows in Training Data = {n_rows_train}, "
        f"Batch Size = {batch_size}") 
    train_losses = []
    val_losses = []
    beta_tc_losses = []
    best_val_loss = float('inf')
    real_epochs = 0

    # Training loop
    for epoch in range(epochs):
        real_epochs += 1
        epoch_loss = 0
        epoch_beta_tc_loss = 0
        for step, batch in enumerate(train_dataset): # window size , features
            global_step = epoch * len(train_dataset) + step  # Total training step count

            anneal_coeff = linear_annealing(0, 1, global_step, steps_anneal)

            # 1. Train VAE
            with tf.GradientTape() as tape:
                reconstructed, mu, logvar = vae(batch, n_samples=n_samples, latent_only = False)  # Use multiple samples

                #Compute reconstruction MSE error for each sample
                reconstruction_errors = tf.reduce_mean(
                    tf.square(tf.expand_dims(batch, axis=0) - reconstructed), axis=-1
                )  # Shape: (n_samples, batch_size, window_size)

                # Aggregate errors (mean over samples and window_size)
                mean_reconstruction_error = tf.reduce_mean(reconstruction_errors, axis=(2, 0))

                # Compute VAE loss
                reconstruction_loss = tf.reduce_mean(mean_reconstruction_error)

                kl_divergence = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
                                
                z = vae.reparameterize(mu, logvar, n_samples=n_samples)
                z = tf.reduce_mean(z, 0) # aggregate over latent samples
                
                tc_beta_loss = vae.b_tcvae_total_correlation_loss(z, mu, logvar)
                b_tcvae_loss = (1 - beta_tc) * tc_beta_loss

                loss = reconstruction_loss + kl_divergence + b_tcvae_loss

            gradients = tape.gradient(loss, vae.encoder.trainable_variables + vae.decoder.trainable_variables 
                                      + vae.fc_mu.trainable_variables + vae.fc_logvar.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimizer.apply_gradients(zip(gradients, vae.encoder.trainable_variables + vae.decoder.trainable_variables 
                                          + vae.fc_mu.trainable_variables + vae.fc_logvar.trainable_variables))

            epoch_loss += loss.numpy()
            epoch_beta_tc_loss += b_tcvae_loss.numpy()
        # VALIDATION
        epoch_val_loss = 0
        show_val = validation_method in ["B_VAE","B_TCVAE"]

        if validation_method == "None" or validation_method == "PLOT":
            val_loss = 0
            early_stop = False
        else:
            early_stop = True
            for batch in val_dataset:
                reconstructed, mu, logvar = vae(batch, n_samples=n_samples, latent_only=False)

                if validation_method == "B_VAE":
                    reconstruction_errors = tf.reduce_mean(
                        tf.square(tf.expand_dims(batch, axis=0) - reconstructed), axis=-1
                    )
                    mean_reconstruction_error = tf.reduce_mean(reconstruction_errors, axis=(2, 0))
                    reconstruction_loss = tf.reduce_mean(mean_reconstruction_error)
                    kl_divergence = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))

                    val_loss = reconstruction_loss + beta * kl_divergence
                    epoch_val_loss += val_loss.numpy()

                elif validation_method == "B_TCVAE":
                    reconstruction_errors = tf.reduce_mean(
                        tf.square(tf.expand_dims(batch, axis=0) - reconstructed), axis=-1
                    )
                    mean_reconstruction_error = tf.reduce_mean(reconstruction_errors, axis=(2, 0))
                    reconstruction_loss = tf.reduce_mean(mean_reconstruction_error)
                    kl_divergence = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))

                    z = vae.reparameterize(mu, logvar, n_samples=n_samples)
                    z = tf.reduce_mean(z, axis= 0)
                    tc_beta_loss = vae.b_tcvae_total_correlation_loss(z, mu, logvar)
                    b_tcvae_loss = (1 - beta_tc) * tc_beta_loss
                    
                    val_loss = reconstruction_loss + kl_divergence + b_tcvae_loss

                    epoch_val_loss += val_loss.numpy()

        if validation_method == "PLOT" and epoch % 50 == 0 and epoch > 0:
            print("PLOT AT EPOCH: ", {epoch + 1})
            get_latent_representations_label(vae, test_dataset, latent_dim, beta,n_critic,gamma,time,epoch = epoch, name = model_name,type = 'TSNE', save = False, AWS = AWS, s3 = s3, BUCKET = BUCKET)

        # Store the loss for this epoch
        train_loss = epoch_loss / len(train_dataset)
        val_loss = epoch_val_loss / len(val_dataset) 
        beta_tc_loss = epoch_beta_tc_loss / len(train_dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        beta_tc_losses.append(beta_tc_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, "
            f"Beta TC Loss: {beta_tc_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if early_stop:
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                # Save best model
                save_trained_model(vae, optimizer, model_path, model_name, latent_dim,beta,n_rows_train,time,epoch, AWS = AWS, s3 = s3 , BUCKET = BUCKET)
                print( f'💽Saved Model at Epoch {epoch+1}💽')
            else:
                wait += 1
                if wait >= patience:
                    epochs = epoch + 1
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
        # Save the trained model
        #vae.compile(optimizer = optimizer)
        #vae.save(model_path)
        if epoch % 50 == 0 and epoch > 0:
            save_trained_model(vae, optimizer, model_path, model_name, latent_dim,beta,n_rows_train,time,epoch, AWS = AWS, s3 = s3 , BUCKET = BUCKET)
            print( f'💽Saved Model at Epoch {epoch+1}💽')

    save_trained_model(vae, optimizer, model_path, model_name, latent_dim,beta,n_rows_train,time,epoch, AWS = AWS, s3 = s3 , BUCKET = BUCKET)
    print( f'💽Saved Model at Epoch {epoch+1}💽')

    print("💽Loss Plot Saved.💽")
    print("💽VAE model saved.💽")
    #vae.summary()
    return train_losses, val_losses, real_epochs, time, show_val, model_path, vae
