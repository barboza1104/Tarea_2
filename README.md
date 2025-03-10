# MNIST Digit Classification with Keras

Neural network implementation for handwritten digit recognition using the MNIST dataset and Keras framework.

## Key Features
- **Dataset**: Uses standard MNIST dataset (60k training/10k test samples)
- **Architecture**: Sequential model with fully connected layers
- **Optimization**: RMSprop optimizer with cross-entropy loss
- **Preprocessing**: Built-in data normalization and label encoding

## Installation
    pip install tensorflow keras numpy


## Usage
From command line:
    python neural_network_keras.py

From another Python script:
    from neural_network_keras import ejecutar_red_neuronal
    ejecutar_red_neuronal()


## Model Architecture
    Sequential([
    Input(shape=(784,)), # Flattened 28x28 images
    Dense(512, activation='relu'),# Hidden layer with ReLU
    Dense(10, activation='softmax') # Output layer with softmax
    ])



## Training Configuration
  - **Epochs**: 8 cycles
  - **Batch Size**: 128 samples
  - **Validation**: Automatic test set evaluation
  - **Metrics**: Accuracy tracking

## Data Pipeline
  1. **Loading**: Auto-downloads MNIST dataset
  2. **Reshaping**: Flattens 28x28 images to 784-dim vectors
  3. **Normalization**: Scales pixel values to [0,1] range
  4. **Label Encoding**: One-hot encoding for categorical crossentropy

## Performance
Model achieves:
  - Training accuracy: ~98% (varies by initialization)
  - Test accuracy: ~97% 
  - Inference speed: <2ms per sample (CPU)

## Dependencies
  - Python 3.7+
  - TensorFlow 2.x
  - Keras 3.x
  - NumPy
