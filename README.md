# HyperEncoder
### Overview
This is an auto-encoder network which uses the architecture from [HyperFace: A Deep Multi-task Learning Framework](https://arxiv.org/abs/1603.01249) to create the embedding and [Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) for the decoder network.
Auto-encoders are a core element in many algorithms, including anomoly detection and DeepFakes. The basic goal of the model is to produce a small *embedding* which retains all the important information from the original image. This is done by creating an *hourglass* shaped neural network. The first half reduces the demensions until it reaches it's smallest size (the embedding dimension). The second half will upscale the embedding and match it back to the original image. 
**Example of Architecture:** below is a visual of the architecture. The compressed image inbetween the encoder and decoder represents the embedding layer. It is roughly the same dimensions as a 100-dim embedding, which is the size used in my experoments.
![alt text](https://github.com/lucaspettit/HyperEncoder/blob/master/images/encode-decode%20example.png)

### Design
This model was trained using the CelebA dataset to produce a 100-dim embedding with the decoder producing a 64x64 RGB image. Mean Squared Error was used to train calculate the loss between the predicted image and the ground truth (original image resized to 64x64). This loss function is different than the one's used by HyperFace or DCGNN since this model has a different target.
Before training the images were cropped to the facial boundingbox region and resized to 227x227 using MTCNN for the facial detection and OpenCV for cropping/resizing.



**Learning Rates & Embedding Stats:** Top-Left: the embedding MAX value, Top-Right: the embedding MIN value, Bottom-Left: the embedding STD, Bottom-Right: the Mean Squared Error loss during training (orange is trining, blue is validation set).


![alt text](https://github.com/lucaspettit/HyperEncoder/blob/master/images/Screenshot%20from%202018-03-13%2008-51-17.png)

### Results
Below are sample images of the faces Ground Truth (labeled *Real*) and the predicted faces from the embeddings.
![alt text](https://github.com/lucaspettit/HyperEncoder/blob/master/images/test_images_epoch_20_face_only.jpg)

### Next Steps
Training was stopped once the model began to overfit the data. I believe more accurate results can come if I retrain on a larger, more diverse dataset. The CelebA data seems to be primarily either people smiling or looing stoic while facing the camera. I've used my other project [GifTheRipper](https://github.com/lucaspettit/GifTheRipper) to collect bulk data with more variance. I intend to retrain this model on the new data soon and am excited to report back the process in more detail.

### Usage

**TRAIN MODEL**
To train the model, run the following command with --dataset_dir pointing to a file with the images to train on. 
You can open up TensorBoard and view training statistics & graphs. 
```
$> python3 main.py --dataset_dir=<path to dataset> --train=True
```

**FLAGS**
Directory Flags
```
--dataset_dir   : The directoy where the dataset is stored.
                   Required=True
--dataset_name  : The name of the dataset. If None, name of dataset directory will be used.
                   Default=None
--checkpoint_dir: Directory name to save the checkpoints.
                   Default=checkpoint
--sample_dir    : Directory name to save image samples.
                   Default=samples
--train         : True for training. False for testing.
                   Default=True
--model_path    : The path for the model to use during testing.
                   Default=models
--tf_record_path: Path for TensorFlow Record Files to use during training.
                   Default=tf_record_path.
```

Training parameters
```
--epoch         : Training epoch. 
                   Default=25
--learning_rate : Learning rate for Adam. 
                   Default=0.0002
--beta1         : Momentum term of Adam.
                   Default=0.5
--batch_size    : The size of the batch images.
                   Default=64
```

Model parameters
```
--embed_size    : Size of the embedding layer.
                   Default=100
--x_height      : The height of the input image.
                   Default=227 (currently only supports 227)
--x_width       : The height of the input image. If None, same value as x_height.
                   Default=None (currently only supports None and 227)
--y_height      : The height of the image to be generated.
                   Defuault=64
--y_width       : The width of the image to be geerated. If None, same value as y_height.
                   Default=None
```