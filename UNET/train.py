import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

def modify_labels_for_unet(y):
    # Modify the labels to have three channels (background, foreground1, foreground2)
    y_modified = np.zeros((y.shape[0], y.shape[1], y.shape[2], 3), dtype=np.float32)
    y_modified[:, :, :, 0] = 1 - y[:, :, :, 0]  # Background channel
    y_modified[:, :, :, 1] = y[:, :, :, 0]      # Foreground channel
    return y_modified

def unet_train(unet, X, y):
    # Split Train and Test Set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=123)

    # # Modify the labels for U-Net
    # y_train = modify_labels_for_unet(y_train)
    # y_valid = modify_labels_for_unet(y_valid)

    # Run the model in a mini-batch fashion and compute the progress for each epoch
    results = unet.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_valid, y_valid))
    return results, unet, X_train, X_valid, y_train, y_valid

