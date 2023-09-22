# UNET on Oxford Pet Dataset
This folder contains the UNET implementation on the [Oxford Pets dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

The implementation is similar to what is done [here](https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406) with a few tweaks. 

We have introduced Dice loss, Binary Cross Entropy Loss, Up Sampling, Max Pooling, Strided Convolution and Transpose Convolution.

U-Net architecture was introduced by Olaf Ronneberger, Philipp Fischer, Thomas Brox in 2015 for tumor detection but since has been found to be useful across multiple industries. As an image segmentation tool, the model aims to classify each pixel as one of the output classes, creating an output resembling fig-1.

Many Neural Nets have tried to perform ‘image segmentation’ before, but U-Net beats its predecessors by being less computationally expensive and minimizing information loss. Let’s deep dive further to learn more about how U-Net does this.

![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/cc78e420-d889-4f50-bb82-30236c4e7238)

UNet, short for "U-shaped Neural Network," is a convolutional neural network (CNN) architecture commonly used in image segmentation tasks, particularly in the field of medical image analysis. It was developed by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in 2015. UNet has gained popularity due to its effectiveness in segmenting objects within images with high accuracy and has been widely adopted in various computer vision applications beyond medical imaging.

Here's a detailed description of the UNet architecture:

1. Encoder-Decoder Structure: UNet is characterized by its U-shaped architecture, which consists of two main parts: the encoder and the decoder. This design allows for the extraction of features through the encoder and then the precise localization of objects through the decoder.

2. Contracting Path (Encoder): The encoder component of UNet is designed to capture context and features from the input image. It typically comprises multiple convolutional layers, often arranged in a downsampling fashion. As you move deeper into the encoder, the spatial dimensions of the feature maps decrease, but the number of channels (feature maps) typically increases. This helps in abstracting and capturing high-level features.

3. Expansive Path (Decoder): The decoder, on the other hand, is responsible for gradually increasing the spatial resolution of the feature maps while maintaining a rich representation of features. It uses transposed convolutional layers or upsampling operations to achieve this. Skip connections are a key feature of UNet that connect corresponding layers from the encoder to the decoder. These skip connections help in preserving fine-grained details and improve the segmentation results.

4. Skip Connections: The skip connections bridge the gap between the encoder and decoder by concatenating or adding feature maps from the encoder to the decoder at each corresponding level. This allows the model to combine both low-level and high-level features, which is crucial for accurate segmentation. Skip connections enable UNet to localize objects precisely.

5. Final Layer: The decoder's output layer typically consists of a single convolutional layer with a sigmoid activation function. This layer generates a probability map where each pixel represents the likelihood of belonging to the target object class. Post-processing techniques such as thresholding or connected component analysis are often applied to produce the final binary segmentation mask.

6. Loss Function: The training of a UNet model involves minimizing a suitable loss function, such as binary cross-entropy or dice coefficient loss, which quantifies the dissimilarity between the predicted segmentation mask and the ground truth mask.

## Modifications
1. Dice Loss: Dice loss, also known as the Sørensen-Dice coefficient, is a similarity coefficient used in image segmentation tasks. It measures the overlap between the predicted and ground truth segmentation masks. By incorporating Dice loss into UNet, you can encourage the model to produce more accurate and precise segmentation results. The Dice loss can replace or complement the traditional binary cross-entropy loss.

2. Binary Cross-Entropy Loss: Binary cross-entropy loss is commonly used in image segmentation tasks and serves as the default loss function in UNet. It measures the dissimilarity between the predicted probability map and the ground truth binary mask. This loss encourages the model to assign high probabilities to pixels corresponding to the target object and low probabilities to background pixels.

2. Up Sampling: Up sampling is an operation used in the decoder part of UNet to increase the spatial resolution of feature maps. Instead of using transposed convolutions, you can employ up sampling techniques like nearest-neighbor or bilinear interpolation followed by a standard convolution layer to upsample feature maps. This helps in preserving features while increasing spatial resolution.

3. Max Pooling: Max pooling is a down-sampling operation that is often used in the encoder portion of UNet to reduce the spatial dimensions of feature maps. Max pooling helps in capturing the most salient features while discarding less relevant information. It can be useful for capturing hierarchical features at different scales.

4. Strided Convolution: Strided convolution is an alternative to max pooling for down-sampling. Instead of pooling layers, you can use convolutional layers with larger strides in the encoder to reduce spatial dimensions. This allows the model to learn which features to keep at each spatial resolution level, potentially providing more flexibility in feature extraction.

5. Transpose Convolution (Deconvolution): Transpose convolution, also known as deconvolution or up-sampling convolution, is commonly used in the decoder part of UNet to increase the spatial resolution of feature maps. It essentially "upscales" feature maps and is crucial for generating high-resolution segmentation masks. Transpose convolution layers can be combined with skip connections to recover fine-grained details and produce accurate segmentations.

By incorporating these components into the UNet architecture, you can achieve a more flexible and powerful model that is capable of handling various image segmentation tasks with improved accuracy and precision. Experimenting with different combinations of these components and loss functions can help tailor UNet to specific applications, making it a versatile choice for a wide range of computer vision tasks.

## Results
1.  Max Pooling +Transpose Conv + Binary Cross-Entropy

    ![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/de7a38e0-65d2-4eaa-820b-120eaf68b95f)

    ```
    Epoch 15/20
    185/185 [==============================] - 41s 221ms/step - loss: 0.0706 - accuracy: 0.7080 - val_loss: 0.0344 - val_accuracy: 0.7588
    Epoch 16/20
    185/185 [==============================] - 41s 222ms/step - loss: 0.0849 - accuracy: 0.7115 - val_loss: 0.1003 - val_accuracy: 0.7085
    Epoch 17/20
    185/185 [==============================] - 41s 221ms/step - loss: 0.1483 - accuracy: 0.7049 - val_loss: 0.1484 - val_accuracy: 0.7085
    Epoch 18/20
    185/185 [==============================] - 41s 221ms/step - loss: 0.0918 - accuracy: 0.7050 - val_loss: 0.1106 - val_accuracy: 0.7085
    Epoch 19/20
    185/185 [==============================] - 41s 221ms/step - loss: 0.0571 - accuracy: 0.7052 - val_loss: -0.0357 - val_accuracy: 0.7093
    Epoch 20/20
    185/185 [==============================] - 41s 221ms/step - loss: 0.0728 - accuracy: 0.7054 - val_loss: 0.1357 - val_accuracy: 0.7085

    ```

2. Max Pooling + Transpose Conv + Dice Loss

    ![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/a164a92d-4af6-4c65-bf97-ce3a8c3b6c76)

    ```
    Epoch 15/20
    185/185 [==============================] - 40s 219ms/step - loss: 1.0412 - accuracy: 0.0112 - val_loss: 1.0634 - val_accuracy: 0.0000e+00
    Epoch 16/20
    185/185 [==============================] - 40s 219ms/step - loss: 1.0375 - accuracy: 0.0000e+00 - val_loss: 0.9851 - val_accuracy: 0.0000e+00
    Epoch 17/20
    185/185 [==============================] - 40s 218ms/step - loss: 0.9014 - accuracy: 4.6272e-05 - val_loss: 0.8231 - val_accuracy: 1.3417e-04
    Epoch 18/20
    185/185 [==============================] - 40s 219ms/step - loss: 0.6964 - accuracy: 0.0190 - val_loss: 0.4729 - val_accuracy: 0.1957
    Epoch 19/20
    185/185 [==============================] - 40s 218ms/step - loss: 0.6331 - accuracy: 0.1607 - val_loss: 0.7076 - val_accuracy: 0.0000e+00
    Epoch 20/20
    185/185 [==============================] - 40s 218ms/step - loss: 0.4898 - accuracy: 0.0786 - val_loss: 0.3791 - val_accuracy: 0.2911
    47/47 [==============================] - 3s 54ms/step - loss: 0.3791 - accuracy: 0.2911
    ```

3. Strided Convolution + Transpose Conv + Binary Cross Entropy

    ![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/7c225c2b-fba1-44ef-a5c4-734cc31567e0)

    ```
    Epoch 15/20
    185/185 [==============================] - 45s 242ms/step - loss: 0.1459 - accuracy: 0.7050 - val_loss: 0.1329 - val_accuracy: 0.7085
    Epoch 16/20
    185/185 [==============================] - 45s 243ms/step - loss: 0.1500 - accuracy: 0.7050 - val_loss: 0.1217 - val_accuracy: 0.7085
    Epoch 17/20
    185/185 [==============================] - 45s 242ms/step - loss: 0.4901 - accuracy: 0.7050 - val_loss: 1.8421 - val_accuracy: 0.7085
    Epoch 18/20
    185/185 [==============================] - 44s 237ms/step - loss: 1.8591 - accuracy: 0.7050 - val_loss: 1.8421 - val_accuracy: 0.7085
    Epoch 19/20
    185/185 [==============================] - 44s 237ms/step - loss: 1.8591 - accuracy: 0.7050 - val_loss: 1.8421 - val_accuracy: 0.7085
    Epoch 20/20
    185/185 [==============================] - 44s 237ms/step - loss: 1.8591 - accuracy: 0.7050 - val_loss: 1.8421 - val_accuracy: 0.7085
    47/47 [==============================] - 3s 58ms/step - loss: 1.8421 - accuracy: 0.7085
    ```

4. Strided Conv + Upssampling + Dice Loss

    ![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/812b6e36-e8aa-445f-9481-ad822f9793a0)

    ```
    Epoch 15/20
    185/185 [==============================] - 47s 256ms/step - loss: 0.6225 - accuracy: 0.7010 - val_loss: 0.5762 - val_accuracy: 0.7085
    Epoch 16/20
    185/185 [==============================] - 47s 256ms/step - loss: 0.4653 - accuracy: 0.3957 - val_loss: 0.3393 - val_accuracy: 0.2915
    Epoch 17/20
    185/185 [==============================] - 47s 256ms/step - loss: 0.9046 - accuracy: 0.2084 - val_loss: 0.2231 - val_accuracy: 0.6910
    Epoch 18/20
    185/185 [==============================] - 47s 255ms/step - loss: 0.6308 - accuracy: 0.7048 - val_loss: 0.9364 - val_accuracy: 0.7085
    Epoch 19/20
    185/185 [==============================] - 47s 256ms/step - loss: 0.7978 - accuracy: 0.7050 - val_loss: 0.6767 - val_accuracy: 0.7085
    Epoch 20/20
    185/185 [==============================] - 47s 256ms/step - loss: 0.4227 - accuracy: 0.5234 - val_loss: -1.1383 - val_accuracy: 0.0000e+00
    47/47 [==============================] - 3s 59ms/step - loss: -1.1383 - accuracy: 0.0000e+00
    ```