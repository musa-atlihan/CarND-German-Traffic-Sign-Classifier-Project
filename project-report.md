# German Traffic Signs Classifier Project

The aim of this project is to classify the German traffic signs with deep convolutional neural network architectures using the [Tensorflow framework](https://www.tensorflow.org/).

## The Dataset

The dataset provided by the [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb) and the details of the full dataset and the results of the final competition was reported by [the paper](http://image.diku.dk/igel/paper/MvCBMLAfTSR.pdf) written by J. Stallkamp and M. Schlipsing and J. Salmen and C. Igel.

The dataset used in this project is the same dataset provided by INI-GTSRB Benchmark with the images are prepared as downloadable pickle file formats by [Udacity self-driving car nanodegree program](https://www.udacity.com/drive). The dataset is available to download [here](https://drive.google.com/open?id=0B8c3GUF4ZQ-_R29QOWxDTkRJV3c).

### The train, validation and the test sets

The dataset is split into three parts as the train, validation and the test sets. The train set consists of 34799 examples, validation set only has 4410 examples and the test set has 12630 examples. The train set is used to train the deep convolutional neural network models while the validation set is used to fine tune the model hyperparameters. After fine tuning and training the models, the test set used to evaluate the model performance.

All the images in these sets has the same dimensions as 32x32 pixels of height and width and 3 (RGB) color channels. The total number of classes of the traffic signs is 43. The traffic sign examples from the same classes was captured in different ligthing conditions, from different viewpoints, some of them had 15x15 input resolution (all the images are resized to 32x32) and some of the examples have physical damage, graffiti and stickers. Some of the images for each class from the test set can be seen below.

|  The Visualization of Some of The Test Set Examples |
|:-------------------------:|
|  ![The test set examples](./images/test-set.png) |

### The distribution of classes in the train, validation and the test sets

It is important to have each class in a dataset has approximately the same number of examples. The graphs below shows the frequency distribution of the classes in train, validation and test sets.

<div>
<img style='float:left; width:50%' src='./images/freq-train.png'>
<img style='float:right; width:50%' src='./images/freq-valid.png'>
<img style='width:50%' align='middle' src='./images/freq-test.png'>
</div>

As we can see from the frequency distributions, in each dataset, the number of samples are different for each class. When this is the case, an overfit may occur when training the model. Thus data augmentation is applied for train and validation sets. In addition, if the trained model guessing at random or most of the time guesses for a class with high frequency, a simple accuracy evaluation may not reflect the truth. Thus, besides the accuracy evaluation, another error metric called [f1-score](https://en.wikipedia.org/wiki/F1_score), where the [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) measures are used together, is also obtained for each class.

### Data augmentation

To avoid overfitting, data augmentation is aplied on train and validation sets. In each set, each class with the number of examples are less than two thousand, rised up to a number of two thousand and ten examples by appying random rotations, translations and viewpoint transformations to existing examples. By having two thousand and ten examples for the train and two thousand examples for the validation sets, the models fine tuned and trained to evaluate on test sets ([link to code](./data-augmentation-equal-numbers.ipynb)).

## The Model Architectures

### A) The LeNet Architecture


![LeNet Architecture](./images/lenet.png)
Source: ([Y. Lecun, L. Bottou, Y. Bengio and P. Haffner, 98](http://ieeexplore.ieee.org/document/726791/))

The augmented dataset, first trained on famous [LeNet-5 architecture](http://ieeexplore.ieee.org/document/726791/). A diagram of this architecture is seen above from Lecun's paper. The architecture consists of two convolutional, two subsampling layers, a flattened layer, two fully connected layers and a final output layer of logits. The input to LeNet-5 is 32x32x1 but since the dataset in this project has 3 color chanels, here the LeNet architecture takes images of 32x32x3 as input dimensions ([link to code](./Traffic_Sign_Classifier.ipynb)).

#### Details of the architecture

As mentioned above, the input images has 32x32x3 dimensions which is different from original LeNet having 3 color dimensions instead of 1. Layer 1 of the model is a convolutional layer having 5x5 kernel size with stride of 1x1, outputs 6 feature maps. After layer 1, a ReLu activation function and a 2x2 max pooling with a stride of 2x2 are applied. Layer 2 is again a convolutional layer with 5x5 kernel size and have 1x1 stride, layer 2 outputs 16 feature maps. After layer 2, ReLu activation function and again a 2x2 max pooling with a stride of 2x2 are applied. Final feature maps are flattened forming a fully connected layer as the layer 3 which has 400 input nodes. The layer 4 is a fully connected layers takes 400 inputs and outputs 120 nodes with ReLu activation function is applied. Layer 5 is again a fully connected layer takes 120 inputs and outputs 80 again with a ReLu activation function applied. The final layer is layer 5 and it takes 84 inputs from the previous layer and, differently from the original LeNet-5, outputs 43 logits for each class in this project.

#### The training and the results for Lenet-5 architecture

The LeNet-5 model tuned and trained on the original training and validation sets without any data augmentation applied. In the training process, stochastic gradient descent algorithm is used with mini-batch size of 128. Apllying Xavier initialization for the weights gave the filexibility to choose larger learning rates ([X. Glorot and Y. Bengio, 2010](http://proceedings.mlr.press/v9/glorot10a.html)). Thus, the learning rate is taken as 0.1 with an exponential decay. An early stopping algorithm is applied and after the training, the test set is evaluated with the best model weights performed on the validation set.

The test accuracy with the best model is 92.243% and the f1-score for each class can be seen below.

