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

### Max Pooling +Transpose Conv + Binary Cross-Entropy

![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/af9ace93-1b49-4127-b088-1a1039829736)

```
Epoch 15/20
185/185 [==============================] - 13s 72ms/step - loss: 0.2302 - accuracy: 0.9126 - val_loss: 0.3247 - val_accuracy: 0.8878
Epoch 16/20
185/185 [==============================] - 13s 72ms/step - loss: 0.2252 - accuracy: 0.9143 - val_loss: 0.3217 - val_accuracy: 0.8817
Epoch 17/20
185/185 [==============================] - 13s 72ms/step - loss: 0.2202 - accuracy: 0.9165 - val_loss: 0.3075 - val_accuracy: 0.8896
Epoch 18/20
185/185 [==============================] - 13s 72ms/step - loss: 0.2124 - accuracy: 0.9193 - val_loss: 0.3120 - val_accuracy: 0.8884
Epoch 19/20
185/185 [==============================] - 13s 72ms/step - loss: 0.2032 - accuracy: 0.9224 - val_loss: 0.3136 - val_accuracy: 0.8870
Epoch 20/20
185/185 [==============================] - 13s 72ms/step - loss: 0.2140 - accuracy: 0.9187 - val_loss: 0.3213 - val_accuracy: 0.8838
```
Sample Prediction:
![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/bd066a27-571b-423f-b080-21a045a45c97)


### Max Pooling + Transpose Conv + Dice Loss

![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/bc3a43ca-2089-4002-9bd7-342ff9918931)

```
Epoch 15/20
185/185 [==============================] - 13s 71ms/step - loss: 0.1719 - accuracy: 0.8892 - val_loss: 0.1936 - val_accuracy: 0.8692
Epoch 16/20
185/185 [==============================] - 13s 71ms/step - loss: 0.1711 - accuracy: 0.8897 - val_loss: 0.1977 - val_accuracy: 0.8672
Epoch 17/20
185/185 [==============================] - 13s 71ms/step - loss: 0.1676 - accuracy: 0.8924 - val_loss: 0.1851 - val_accuracy: 0.8782
Epoch 18/20
185/185 [==============================] - 13s 71ms/step - loss: 0.1609 - accuracy: 0.8981 - val_loss: 0.1811 - val_accuracy: 0.8806
Epoch 19/20
185/185 [==============================] - 13s 71ms/step - loss: 0.1618 - accuracy: 0.8972 - val_loss: 0.1788 - val_accuracy: 0.8836
Epoch 20/20
185/185 [==============================] - 13s 71ms/step - loss: 0.1559 - accuracy: 0.9018 - val_loss: 0.1734 - val_accuracy: 0.8871
```
Sample prediction:
![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/26f803cd-7859-4b5a-92f5-52f868f168b6)

### Strided Convolution + Transpose Conv + Binary Cross Entropy
![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/5c9b8f15-1f8c-4bc1-a279-e8fcd7203827)

```
Epoch 15/20
185/185 [==============================] - 14s 78ms/step - loss: 0.2582 - accuracy: 0.9024 - val_loss: 0.3331 - val_accuracy: 0.8783
Epoch 16/20
185/185 [==============================] - 14s 78ms/step - loss: 0.2364 - accuracy: 0.9102 - val_loss: 0.3467 - val_accuracy: 0.8802
Epoch 17/20
185/185 [==============================] - 14s 78ms/step - loss: 0.2301 - accuracy: 0.9127 - val_loss: 0.3507 - val_accuracy: 0.8762
Epoch 18/20
185/185 [==============================] - 14s 78ms/step - loss: 0.2245 - accuracy: 0.9147 - val_loss: 0.3381 - val_accuracy: 0.8819
Epoch 19/20
185/185 [==============================] - 14s 78ms/step - loss: 0.2109 - accuracy: 0.9195 - val_loss: 0.3219 - val_accuracy: 0.8857
Epoch 20/20
185/185 [==============================] - 14s 78ms/step - loss: 0.2129 - accuracy: 0.9190 - val_loss: 0.3423 - val_accuracy: 0.8850
```
Sample Prediction:
![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/07340570-f487-416c-a90d-1af75c7e6e16)

### Strided Conv + Up Sampling + Dice Loss

![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/f7e8b89b-5aee-43c6-8f04-fd4b02bc211b)

```
Epoch 15/20
185/185 [==============================] - 15s 81ms/step - loss: 0.7523 - accuracy: 0.5919 - val_loss: 0.7509 - val_accuracy: 0.5972
Epoch 16/20
185/185 [==============================] - 15s 81ms/step - loss: 0.7523 - accuracy: 0.5919 - val_loss: 0.7509 - val_accuracy: 0.5972
Epoch 17/20
185/185 [==============================] - 15s 81ms/step - loss: 0.7523 - accuracy: 0.5919 - val_loss: 0.7509 - val_accuracy: 0.5972
Epoch 18/20
185/185 [==============================] - 15s 81ms/step - loss: 0.7523 - accuracy: 0.5919 - val_loss: 0.7509 - val_accuracy: 0.5972
Epoch 19/20
185/185 [==============================] - 15s 81ms/step - loss: 0.7523 - accuracy: 0.5919 - val_loss: 0.7509 - val_accuracy: 0.5972
Epoch 20/20
185/185 [==============================] - 15s 81ms/step - loss: 0.7523 - accuracy: 0.5919 - val_loss: 0.7509 - val_accuracy: 0.5972
```
Sample Prediction:
![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/cf33d6c1-1938-456d-b916-e75393a64fea)

## Conclusions
From the above logs we can see that the first three strategies worked really well for us and the last one did not meet expectations. The Max Pooling +Transpose Conv + Binary Cross-Entropy strategy seems to have the highest accuracy and comparing the sample prediction it seems to have a slight edge over the Max Pooling + Transpose Conv + Dice Loss and Strided Convolution + Transpose Conv + Binary Cross Entropy strategies. 
 