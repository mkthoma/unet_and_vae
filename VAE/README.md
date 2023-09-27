# VAE on MNIST and CIFAR10

The folder contains the implementation of VAE on the MNIST and CIFAR10 datasets.

In the last few years, deep learning based generative models have gained more and more interest due to (and implying) some amazing improvements in the field. Relying on huge amount of data, well-designed networks architectures and smart training techniques, deep generative models have shown an incredible ability to produce highly realistic pieces of content of various kind, such as images, texts and sounds. Among these deep generative models, two major families stand out and deserve a special attention: Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs).

Variational Autoencoders (VAEs) are generative models explicitly designed to capture the underlying probability distribution of a given dataset and generate novel samples. They utilize an architecture that comprises an encoder-decoder structure. The encoder transforms input data into a latent form, and the decoder aims to reconstruct the original data based on this latent representation. The VAE is programmed to minimize the dissimilarity between the original and reconstructed data, enabling it to comprehend the underlying data distribution and generate new samples that conform to the same distribution.

One notable advantage of VAEs is their ability to generate new data samples resembling the training data. Because the VAE’s latent space is continuous, the decoder can generate new data points that seamlessly interpolate among the training data points. VAEs find applications in various domains like density estimation and text generation.

Variational autoencoder is different from autoencoder in a way such that it provides a statistic manner for describing the samples of the dataset in latent space. Therefore, in variational autoencoder, the encoder outputs a probability distribution in the bottleneck layer instead of a single output value.

![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/d68e8e2b-a0ed-4abd-8a34-16a0837bcb75)

The two notebooks attached in the folder have implementation on how to run the VAE for MNIST and CIFAR10 datasets.

For both the notebooks, we have tried testing the model's prediction on correctly labelled data and incorrectly labelled data.

### MNIST
#### Correctly labelled data
![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/b8cf9a75-74a2-47b1-a49b-002063205f20)

#### Incorrectly labelled data
![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/5b5172ac-5ab8-43bb-a162-bdac02c536a1)

### CIFAR10
#### Correctly labelled data
![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/8b45e0ce-49fd-4ce9-bffb-a4d62c44fe19)

#### Incorrectly labelled data
![image](https://github.com/mkthoma/unet_and_vae/assets/135134412/73208cda-578e-4d10-bcec-67bbbe2e68c2)
