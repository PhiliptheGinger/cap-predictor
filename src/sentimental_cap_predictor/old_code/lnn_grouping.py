import torch
import torch.nn as nn
import pandas as pd
from loguru import logger

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout_rate=0.0, bidirectional=False):
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        
        if self.num_layers == 1:
            dropout_rate = 0.0
        dropout_rate = float(dropout_rate)

        self.recurrent_layer = nn.LSTM(int(input_dim), self.hidden_dim, num_layers=self.num_layers, 
                                       dropout=dropout_rate, batch_first=True, bidirectional=self.bidirectional)
        direction_multiplier = 2 if self.bidirectional else 1
        
        self.fc_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dim * direction_multiplier, int(output_dim))
        )
        
        self.initialize_weights()  # Initialize weights

        logger.debug(f"LiquidNeuralNetwork initialized with input_dim: {input_dim}, hidden_dim: {self.hidden_dim}, output_dim: {output_dim}, num_layers: {self.num_layers}, dropout_rate: {dropout_rate}, bidirectional: {self.bidirectional}")

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LSTM):
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        logger.debug(f"Input tensor shape: {x.shape}")

        if not x.dtype == torch.float32:
            x = x.float()
            logger.debug("Converted input tensor to float32.")

        if torch.isnan(x).any():
            logger.warning("NaN values found in input tensor. Forward-filling NaNs.")
            x = self.forward_fill_nan(x)

        x = torch.nan_to_num(x)
        
        x, _ = self.recurrent_layer(x)
        logger.debug(f"Output tensor shape after LSTM: {x.shape}")
        
        x = x[:, -1, :]
        
        x = self.fc_layer(x)
        logger.debug(f"Output tensor shape after fully connected layer: {x.shape}")
        
        return x

    def forward_fill_nan(self, x):
        for i in range(1, x.size(1)):
            nan_mask = torch.isnan(x[:, i, :])
            x[:, i, :][nan_mask] = x[:, i - 1, :][nan_mask]
        return x

def train_liquid_nn(model, optimizer, criterion, X_train, y_train, epochs=100, patience=10, clip_value=1.0):
    best_loss = float('inf')
    patience_counter = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(epochs):
        model.train()

        predictions = model(X_train)

        if y_train.shape != predictions.shape:
            y_train = y_train.view(predictions.shape)

        loss = criterion(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        scheduler.step(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs. Best loss: {best_loss}")
            break

        logger.info(f"Epoch {epoch + 1}/{epochs}: Loss = {loss.item()}")

    return model

def create_liquid_ensemble_predictions(df, selected_models, best_params):
    logger.debug(f"Creating Liquid Ensemble predictions with parameters: {best_params}")
    
    features = selected_models + ['day_of_week', 'month', 'day_of_month', 'quarter', 'year'] + \
               [f'fourier_sin_{i}' for i in range(1, 3)] + \
               [f'fourier_cos_{i}' for i in range(1, 3)]
    
    df = df[features].dropna()

    X = df.values
    logger.debug(f"Original DataFrame shape: {df.shape}")
    logger.debug(f"Reshaped tensor shape: {X.shape}")

    X = X.astype('float32')

    X = torch.tensor(X.reshape(X.shape[0], 1, X.shape[1]), dtype=torch.float32)
    logger.debug(f"Tensor shape after reshaping: {X.shape}")

    output_dim = 14

    logger.debug(f"Inferred output_dim: {output_dim}")

    model = LiquidNeuralNetwork(
        input_dim=int(X.shape[2]), 
        hidden_dim=int(best_params['hidden_dim']), 
        output_dim=output_dim, 
        num_layers=int(best_params['num_layers']), 
        dropout_rate=float(best_params['dropout_rate']), 
        bidirectional=bool(best_params['bidirectional'])
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(best_params['learning_rate']))
    criterion = nn.MSELoss()

    model = train_liquid_nn(model, optimizer, criterion, X, X, epochs=int(best_params['epochs']))

    model.eval()
    with torch.no_grad():
        predictions = model(X)
        predictions = predictions.cpu().numpy()

    return pd.DataFrame(predictions, index=df.index)
