# HyperEncoder
### Overview
This is an auto-encoder network which uses the architecture from [HyperFace: A Deep Multi-task Learning Framework](https://arxiv.org/abs/1603.01249) to create the embedding and [Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) for the decoder network.
Auto-encoders are a core element in many algorithms, including anomoly detection and DeepFakes. The basic goal of the model is to produce a small *embedding* which retains all the important information from the original image. This is done by creating an hourglass shaped neural network. The first half reduces the demensions until it reaches it's smallest size (the embedding dimension) which is a 1xN dimentional vector. The second half will upscale the embedding and match it back to the original image. 
**Example of Architecture:** below is a visual of the architecture. The compressed image inbetween the encoder and decoder represents the embedding layer. The 7x7 pixel image contains approximately the same amount of information as the 128-dimensional vector used in this auto-encoder's trained model.
![alt text](https://github.com/lucaspettit/HyperEncoder/blob/master/images/encode-decode%20example.png)

### Model Design
This model uses a trunctated AlixNet as described in [HyperFace's paper](https://arxiv.org/abs/1603.01249) to transform a 227x227 RGB image into a 128-dim vector with the decoder producing a 128x128 RGB image from that embedding vector. However, the embedding vector size can easily be changed to any integer value greater than 0 and less than 512.
To train the model, the Mean SquaredError loss function was evaluated to provide the best trainig results for this perticular system. 

### Data Set and Preprocessing
Multiple publically avialable data set (including CelebA, LFW, and WIDER), as well as about 100,000 images download off the internet were used to train this model. All the preprocessing was done through with [GifTheRipper's](https://github.com/lucaspettit/GifTheRipper) *silverDataScrub* and *goldDataScrub* code. The preprocessing includes detecting faces and landmarks with the [MTCNN algorithm](https://github.com/pangyupo/mxnet_mtcnn_face_detection). The boundingbox for each face was then centered on the nose using the landmark points, and finally resized to 227x227.

### Results
Below are samples of facial images captured from GIF's. Sample images were not used during the training phase. As such, they are all previously unseen to the model. 
This first set of images is from an earlier build which was trained trained on a smaller dataset and produced 64x64 RGB images. The Ground Truth (labeled *Real*) is in the left column, and the decoded faces from the embeddings are in the right column.
![alt text](https://github.com/lucaspettit/HyperEncoder/blob/master/images/test_images_epoch_20_face_only.jpg)

This second example is from the [HyperFace 3de128o128_0.0.1](https://github.com/lucaspettit/HyperEncoder/releases) release which was trained on the full concatenated dataset. This model uses a 128-dimentional embedding and decodes into a 128x128 RGB image. 

![alt text](https://github.com/lucaspettit/HyperEncoder/raw/master/output/4.gif)
![alt text](https://github.com/lucaspettit/HyperEncoder/raw/master/output/2.gif)
![alt text](https://github.com/lucaspettit/HyperEncoder/raw/master/output/5.gif)

### Next Steps
Longer training time will produce better quality results. The best release to date ([3de128o128_0.0.1](https://github.com/lucaspettit/HyperEncoder/releases)
lucaspettit/GifTheRipper)) was still showing progress when stopped. 
I am currently considering integrating [GifTheRipper's](https://github.com/lucaspettit/GifTheRipper) data scrubbing procedures into this repo to provide a smooth flow to capture faces and preprocess them in the method required by HyperFace to produce the highest quality embeddings. 

### Trained Models
1. [HyperFace_3d_e128_o128](https://github.com/lucaspettit/HyperEncoder/releases/download/3de128o128_0.0.1/allFaces_3d_e128_o128.pb). This model produces the highest quality decoded outputs from a 128-dimentional embedding. 
2. [HyperFace_3d_e64_o128](https://github.com/lucaspettit/HyperEncoder/releases/download/3de64o128-0.0.1/hyperencoder_i3d_e64_o128.pb). This model produces a 64-dimentional embedding. 

### Usage
**CONFIG JSON**
The *main.py* file takes the path to the config JSON as an argument. A sample of the config JSON is provided in the config/ directory. 
