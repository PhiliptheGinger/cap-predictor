import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(input_shape, learning_rate=0.001):
    """
    Builds a deep learning model using LSTM layers.
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

def calculate_learning_curve(X_train, y_train, X_val, y_val, batch_size=32, epochs=50, train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Calculates the learning curve for the deep learning model and returns it as a DataFrame.
    
    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation target.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
        train_sizes (np.ndarray): Array of training set proportions.
        
    Returns:
        df_learning_curve (pd.DataFrame): DataFrame containing train sizes, train losses, and val losses.
    """
    learning_curve_data = []

    for train_size in train_sizes:
        # Subset the training data
        subset_size = int(train_size * X_train.shape[0])
        X_train_subset = X_train[:subset_size]
        y_train_subset = y_train[:subset_size]
        
        # Build and train the model on the subset
        model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        history = model.fit(X_train_subset, y_train_subset, 
                            validation_data=(X_val, y_val), 
                            batch_size=batch_size, 
                            epochs=epochs, 
                            verbose=0)
        
        # Record the final training and validation losses
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        
        # Append the data to the list
        learning_curve_data.append({
            'Train Size': subset_size,
            'Train Loss': train_loss,
            'Validation Loss': val_loss
        })
        
        print(f"Train size: {subset_size}, Train loss: {train_loss}, Val loss: {val_loss}")
    
    # Convert the learning curve data into a DataFrame
    df_learning_curve = pd.DataFrame(learning_curve_data)
    
    return df_learning_curve

if __name__ == "__main__":
    # Load the preprocessed data (you may need to adjust the paths)
    X_train = np.load('./preprocessed_data/X_train.npy')
    X_val = np.load('./preprocessed_data/X_val.npy')
    y_train = np.load('./preprocessed_data/y_train.npy')
    y_val = np.load('./preprocessed_data/y_val.npy')
    
    # Reshape data for LSTM input (samples, timesteps, features)
    timesteps = 1
    X_train = np.reshape(X_train, (X_train.shape[0], timesteps, X_train.shape[1]))
    X_val = np.reshape(X_val, (X_val.shape[0], timesteps, X_val.shape[1]))

    # Calculate the learning curve
    train_sizes = np.linspace(0.1, 1.0, 10)  # 10 different training sizes
    df_learning_curve = calculate_learning_curve(X_train, y_train, X_val, y_val, train_sizes=train_sizes)

    # Display the DataFrame containing the learning curve
    print(df_learning_curve)
