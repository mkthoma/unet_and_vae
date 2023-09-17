import numpy as np # for using np arrays
from .dataset import *
from .model import *
from .utils import *
from .train import *

# Load and View Data
""" Load Train Set and view some examples """
# Call the apt function
path1 = '/content/drive/My Drive/U-NET - Implementation/images/original/'
path2 = '/content/drive/My Drive/U-NET - Implementation/images/masks/'
img, mask = LoadData (path1, path2)

show_sample_images(path1, path2, img, mask, show_images = 1)

# Process Data
# Define the desired shape
target_shape_img = [128, 128, 3]
target_shape_mask = [128, 128, 1]

# Process data using apt helper function
X, y = PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2)

# QC the shape of output and classes in output dataset 
print("X Shape:", X.shape)
print("Y shape:", y.shape)
# There are 3 classes : background, pet, outline 
rint(np.unique(y))

show_processed_image(X, y, image_index = 0)

# Train the model
results, unet, X_train, X_valid, y_train, y_valid = unet_train(UNetCompiled, X, y)

# Evaluate Model Results
model_metrics(results)

# RESULTS
# The train loss is consistently decreasing showing that Adam is able to optimize the model and find the minima
# The accuracy of train and validation is ~90% which is high enough, so low bias
# and the %s aren't that far apart, hence low variance

# View Predicted Segmentations
unet.evaluate(X_valid, y_valid)

# Add any index to contrast the predicted mask with actual mask
index = 700
VisualizeResults(X_valid, unet, y_valid, index)