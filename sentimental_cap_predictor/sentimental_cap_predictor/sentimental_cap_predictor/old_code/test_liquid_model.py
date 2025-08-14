import numpy as np
import pytest
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from .time_series_deep_learner import LiquidLayer, build_liquid_model, create_rolling_window_sequences

def test_liquid_layer_initialization():
    # Test that the LiquidLayer initializes correctly with the given number of units
    layer = LiquidLayer(64)
    assert layer.units == 64
    assert layer.state_size == 64

def test_liquid_layer_build():
    # Test the build function of LiquidLayer
    layer = LiquidLayer(64)
    layer.build((None, 10))  # Assume input shape is (None, 10)
    assert layer.kernel.shape == (10, 64)
    assert layer.recurrent_kernel.shape == (64, 64)
    assert layer.bias.shape == (64,)

def test_liquid_layer_call():
    # Test the call function of LiquidLayer
    layer = LiquidLayer(64)
    layer.build((None, 10))  # Assume input shape is (None, 10)
    inputs = np.random.random((1, 10)).astype(np.float32)
    states = [np.random.random((1, 64)).astype(np.float32)]
    output, new_states = layer.call(inputs, states)
    
    assert output.shape == (1, 64)
    assert new_states[0].shape == (1, 64)

def test_build_liquid_model():
    # Test the build_liquid_model function
    model = build_liquid_model((10, 1))
    assert isinstance(model, Sequential)
    assert len(model.layers) == 6
    assert isinstance(model.layers[0], layers.RNN)
    assert isinstance(model.layers[1], layers.Dropout)
    assert isinstance(model.layers[2], layers.RNN)
    assert isinstance(model.layers[3], layers.Dropout)
    assert isinstance(model.layers[4], layers.Dense)

def test_training_process():
    # Test the training process to ensure no errors during training
    X, y = create_rolling_window_sequences(np.random.random(100), 10)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = build_liquid_model((10, 1))
    
    try:
        model.fit(X, y, epochs=1, batch_size=10)
        success = True
    except Exception as e:
        print(f"Training failed with error: {e}")
        success = False
    
    assert success, "Model training failed."

def test_model_prediction():
    # Test the prediction process to ensure the model can predict without errors
    X, y = create_rolling_window_sequences(np.random.random(100), 10)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = build_liquid_model((10, 1))
    model.fit(X, y, epochs=1, batch_size=10)
    
    try:
        predictions = model.predict(X)
        assert predictions.shape[0] == X.shape[0], "Prediction shape mismatch"
        success = True
    except Exception as e:
        print(f"Prediction failed with error: {e}")
        success = False
    
    assert success, "Model prediction failed."

def test_model_cloning():
    # Test the cloning process to ensure the model can be cloned without errors
    model = build_liquid_model((10, 1))

    try:
        # Manually clone the model
        model_clone = build_liquid_model((10, 1))
        model_clone.set_weights(model.get_weights())
        model_clone.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        success = True
    except Exception as e:
        print(f"Model cloning failed with error: {e}")
        success = False

    assert success, "Model cloning failed."

if __name__ == "__main__":
    pytest.main()
