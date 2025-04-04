import tensorflow as tf   
import numpy as np

class ArbitrationIDEmbedding(tf.keras.Model):
    def __init__(self, num_ids, embedding_dim):
        super(ArbitrationIDEmbedding, self).__init__()
        
        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=num_ids, 
            output_dim=embedding_dim
        )
        
        # Prediction head (optional)
        self.predictor = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim // 2, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    
    def call(self, id_indices):
        # Get embeddings
        embeddings = self.embedding(id_indices)
        
        # Optional prediction
        prediction = self.predictor(embeddings)
        
        return embeddings, prediction
    

def min_max_normalize(embeddings):
    """
    Applies Min-Max Normalization (scales embeddings to range [0,1])
    """
    min_val = tf.reduce_min(embeddings, axis=0, keepdims=True)
    max_val = tf.reduce_max(embeddings, axis=0, keepdims=True)
    return (embeddings - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero

def z_score_normalize(embeddings):
    """
    Applies Z-score normalization (mean=0, std=1)
    """
    mean = tf.reduce_mean(embeddings, axis=0, keepdims=True)
    std = tf.math.reduce_std(embeddings, axis=0, keepdims=True)
    z_normalized = (embeddings - mean) / (std + 1e-8)  # Avoid division by zero
    return z_normalized.numpy().tolist()

def train_embedding(unique_ids, num_unique_ids, labels=None, num_epochs=100, embedding_dim=10):
    # Create continuous indices for unique IDs
    id_indices = tf.range(num_unique_ids)
    
    # Create labels if not provided
    if labels is None:
        labels = tf.random.normal((num_unique_ids, 1))
    else:
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    
    # Initialize model
    model = ArbitrationIDEmbedding(
        num_ids=num_unique_ids, 
        embedding_dim=embedding_dim
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.MeanSquaredError()
    )
    
    print("---Training Embedding Network for Arbitration IDs---")
    # Custom training loop
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            # Forward pass
            embeddings, predictions = model(id_indices)
            
            # Compute loss
            loss = tf.reduce_mean(tf.square(labels - predictions))
        
        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Apply gradients
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Print progress
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {tf.reduce_mean(loss).numpy():.4f}')
    
    # Extract learned embeddings
    learned_embeddings = model.embedding.get_weights()[0]
    normalized_embeddings = z_score_normalize(learned_embeddings)
    print("-- ------------------ ---")

    return normalized_embeddings, model