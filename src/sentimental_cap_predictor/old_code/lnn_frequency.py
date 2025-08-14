import torch
import torch.nn as nn
import pandas as pd
from loguru import logger

# Define the SeasonalLSTM class
class SeasonalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout_rate=0.0, bidirectional=False):
        super(SeasonalLSTM, self).__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        
        if self.num_layers == 1:
            dropout_rate = 0.0
        dropout_rate = float(dropout_rate)
        
        # LSTM layer to capture temporal dependencies and seasonality
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, num_layers=self.num_layers, 
                            dropout=dropout_rate, batch_first=True, bidirectional=self.bidirectional)
        direction_multiplier = 2 if self.bidirectional else 1
        
        # Fully connected layer for final output
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dim * direction_multiplier, output_dim)
        )
        logger.debug(f"SeasonalLSTM initialized with input_dim: {input_dim}, hidden_dim: {self.hidden_dim}, output_dim: {output_dim}, num_layers: {self.num_layers}, dropout_rate: {dropout_rate}, bidirectional: {self.bidirectional}")

    def forward(self, x):
        logger.debug(f"Input tensor shape: {x.shape}")

        if not x.dtype == torch.float32:
            x = x.float()
            logger.debug("Converted input tensor to float32.")
        
        if torch.isnan(x).any():
            logger.warning("NaN values found in input tensor. Forward-filling NaNs.")
            x = self.forward_fill_nan(x)
        x = torch.nan_to_num(x)
        
        # LSTM forward pass
        x, _ = self.lstm(x)
        logger.debug(f"Output tensor shape after LSTM: {x.shape}")
        
        # Use the output of the last time step for prediction
        x = x[:, -1, :]  # Shape: (batch_size, hidden_dim * direction_multiplier)
        
        # Fully connected layer forward pass
        x = self.fc(x)
        logger.debug(f"Output tensor shape after fully connected layer: {x.shape}")
        
        return x

    def forward_fill_nan(self, x):
        for i in range(1, x.size(1)):
            nan_mask = torch.isnan(x[:, i, :])
            x[:, i, :][nan_mask] = x[:, i - 1, :][nan_mask]
        return x

# Updated train_seasonal_lstm function
def train_seasonal_lstm(model, optimizer, criterion, X_train, y_train, X_val=None, y_val=None, epochs=100, patience=10, save_path="best_model.pth"):
    best_loss = float('inf')
    patience_counter = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    if X_val is not None and y_val is not None:
        X_val, y_val = X_val.to(device), y_val.to(device)

    for epoch in range(epochs):
        model.train()
        
        predictions = model(X_train)

        # Adjust predictions to match y_train shape dynamically
        if predictions.shape[1] != y_train.shape[1]:
            logger.warning(f"Adjusting predictions shape from {predictions.shape} to match y_train shape {y_train.shape}")
            predictions = predictions[:, :y_train.shape[1], :]

        # Reshape predictions to match y_train shape
        predictions = predictions.view_as(y_train)

        # Ensure shapes match before calculating loss
        assert y_train.shape == predictions.shape, f"Shape mismatch: y_train shape {y_train.shape}, predictions shape {predictions.shape}"

        loss = criterion(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if X_val is not None and y_val is not None:
                val_predictions = model(X_val)
                
                # Adjust validation predictions to match y_val shape dynamically
                if val_predictions.shape[1] != y_val.shape[1]:
                    logger.warning(f"Adjusting val_predictions shape from {val_predictions.shape} to match y_val shape {y_val.shape}")
                    val_predictions = val_predictions[:, :y_val.shape[1], :]

                val_predictions = val_predictions.view_as(y_val)

                assert y_val.shape == val_predictions.shape, f"Shape mismatch: y_val shape {y_val.shape}, val_predictions shape {val_predictions.shape}"

                val_loss = criterion(val_predictions, y_val).item()
            else:
                val_loss = loss.item()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"Epoch {epoch + 1}/{epochs}: Training Loss = {loss.item()}, Validation Loss = {val_loss} (Best)")
        else:
            patience_counter += 1
            logger.info(f"Epoch {epoch + 1}/{epochs}: Training Loss = {loss.item()}, Validation Loss = {val_loss}")

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs. Best Validation Loss: {best_loss}")
            break

    return model

# Function to create predictions using the SeasonalLSTM model
def create_seasonal_predictions(df, selected_models, best_params):
    logger.debug(f"Creating Seasonal LSTM predictions with parameters: {best_params}")
    
    # Use raw time-series data and time-based features
    features = selected_models + ['day_of_week', 'month', 'day_of_month', 'quarter', 'year']  # Modify as needed
    
    df = df[features].dropna()

    X = df.values
    logger.debug(f"Original DataFrame shape: {df.shape}")
    logger.debug(f"Reshaped tensor shape: {X.shape}")

    X = X.astype('float32')

    # Reshape to match LSTM input format (batch_size, sequence_length, input_dim)
    X = torch.tensor(X.reshape(X.shape[0], 1, X.shape[1]), dtype=torch.float32)
    logger.debug(f"Tensor shape after reshaping: {X.shape}")

    # Determine output_dim based on your problem context
    output_dim = 14

    logger.debug(f"Inferred output_dim: {output_dim}")

    # Initialize the model with the best parameters
    model = SeasonalLSTM(input_dim=int(X.shape[2]), hidden_dim=int(best_params['hidden_dim']), 
                         output_dim=output_dim, num_layers=int(best_params['num_layers']), 
                         dropout_rate=float(best_params['dropout_rate']), 
                         bidirectional=bool(best_params['bidirectional']))

    optimizer = torch.optim.Adam(model.parameters(), lr=float(best_params['learning_rate']))
    criterion = nn.MSELoss()

    # Train the model
    model = train_seasonal_lstm(model, optimizer, criterion, X, X, epochs=int(best_params['epochs']))

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        predictions = predictions.numpy()

    return pd.DataFrame(predictions, index=df.index)
