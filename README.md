
# HyperEncoder
### Overview
This is an auto-encoder network which uses the arcitecture from [HyperFace: A Deep Multi-task Learning Framework](https://arxiv.org/abs/1603.01249) to create the embedding and [Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) for the decoder network.
Auto-encoder's are a core element of many algorithms, including anomoly detection and DeepFakes. The basic goal of an auto-encoder is to generate an *embedding* which retains all the important information from the original image. 

![alt text](https://github.com/lucaspettit/HyperEncoder/tree/master/images/encode-decode example.png)

### Design
explain where the base models came from. explain how they were stitched together. explain the loss function i had to use to make this work. (also explain the TensorBoard output)

### Results
show the result graphs from tensorboard and the example images. Explain a little about the embedding layer's histogram and how it might be better to use a smaller embedding layer.

![alt text](https://github.com/lucaspettit/HyperEncoder/tree/master/images/Screenshot from 2018-03-13 08-51-17.png)

![alt text](https://github.com/lucaspettit/HyperEncoder/tree/master/images/test_images_epoch_20_face_only.jpg)



### Usage