import tensorflow as tf
from sklearn.model_selection import train_test_split

def unet_train(UNetCompiled, X, y):
    # Split Train and Test Set
    # Use scikit-learn's function to split the dataset
    # 20% data as test/valid set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=123)

    # Build U-Net Architecture
    # Call the helper function for defining the layers for the model, given the input image size
    unet = UNetCompiled(input_size=(128,128,3), n_filters=32, n_classes=3)
    # Check the summary to better interpret how the output dimensions change in each layer
    unet.summary()


    # Compile and Run Model
    # There are multiple optimizers, loss functions and metrics that can be used to compile multi-class segmentation models
    # Ideally, try different options to get the best accuracy
    unet.compile(optimizer=tf.keras.optimizers.Adam(), 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    # Run the model in a mini-batch fashion and compute the progress for each epoch
    results = unet.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_valid, y_valid))
    return results, unet, X_train, X_valid, y_train, y_valid
