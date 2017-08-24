# German Traffic Signs Classifier Project

The aim of this project is to classify the German traffic signs implementing deep convolutional neural network architectures with the [Tensorflow framework](https://www.tensorflow.org/).

## The Dataset

The dataset is provided by [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb). The details of the dataset and the results of the final competition is reported on [this paper](http://image.diku.dk/igel/paper/MvCBMLAfTSR.pdf) written by J. Stallkamp, M. Schlipsing, J. Salmen and C. Igel.

The dataset used in this project is the same dataset provided by INI-GTSRB Benchmark with the images are prepared in downloadable pickle file formats by [Udacity self-driving car nanodegree program](https://www.udacity.com/drive) and is available [here](https://drive.google.com/open?id=0B8c3GUF4ZQ-_R29QOWxDTkRJV3c).

### The train, validation and the test sets

The dataset is split into three parts as the train, validation and the test sets. The train set consists of 34799 examples, validation set only has 4410 examples and the test set has 12630 examples. The train set is used for training the deep convolutional neural network models while the validation set is used to fine tune the model hyperparameters. After fine tuning and training the models, the test set used for evaluating the model.

All the images in these sets has the same dimensions of 32x32 pixels of height and width and 3 RGB color channels. The total number of classes of the traffic signs is 43. The traffic sign examples from the same classes was captured in different ligthing conditions, from different viewpoints, some of them have low, even 15x15 input resolution (all the images are resized to 32x32) and some of the examples have physical damage, graffiti and stickers. The randomly choosen images for each class from the test set can be seen below.

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

As seen from the frequency distributions, in each dataset, the number of examples are different for each class. When this is the case, an overfit may occur while training the model. Thus, for the final network model ([scaled AlexNet](#c-the-alexnet-architecture)), data augmentation will be applied for train and validation sets. In addition, if the trained model guessing at random or most of the time guesses for a class with high frequency, a simple accuracy evaluation may not reflect the truth. Thus, besides the accuracy evaluation, another error metric called [f1-score](https://en.wikipedia.org/wiki/F1_score), where the [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) measures are used together, is also obtained for each class.


## The Model Architectures

### A) The LeNet Architecture

![LeNet Architecture](./images/lenet.png)
Source: ([Y. Lecun, L. Bottou, Y. Bengio and P. Haffner, 98](http://ieeexplore.ieee.org/document/726791/))

At first, without any data augmentation, the famous [LeNet-5 architecture](http://ieeexplore.ieee.org/document/726791/) is used for training. A diagram of this architecture is shown above from Lecun's [paper]((http://ieeexplore.ieee.org/document/726791/)). The architecture consists of two convolutional, two subsampling layers, a flattened layer, two fully connected layers and a final output layer of logits. The input to original LeNet-5 is 32x32x1 but since the dataset in this project has 3 color chanels, the LeNet architecture takes images of 32x32x3 here. For the implementation details, please visit [this notebook](./Traffic_Sign_Classifier.ipynb).

#### Details of the architecture

As mentioned above, the input images has 32x32x3 dimensions which is different from original LeNet by having 3 color dimensions instead of 1. Layer 1 of the model is a convolutional layer having 5x5 kernel size with stride of 1x1 and outputs 6 feature maps. After layer 1, a ReLu activation function and a 2x2 max pooling with a stride of 2x2 are applied. Layer 2 is again a convolutional layer with 5x5 kernel size and have 1x1 stride, layer 2 outputs 16 feature maps. And again, ReLu activation function and a 2x2 max pooling with a stride of 2x2 are applied. After these operations, the feature maps are flattened to form the layer 3 which is fully connected to the following layer. Layer 3 has 400 input and 120 output connections. The layer 4 is a fully connected layer takes 120 inputs and outputs 84 with ReLu activation function is applied. The layer 5 is the final layer and it has 84 connections as the input from the previous layer and, differently from the original LeNet-5, outputs 43 logits that is one for each class.

#### The training and the results for Lenet-5 architecture

The LeNet-5 model tuned and trained on the original training and validation sets without any data augmentation applied. In the [training process](./trainer.py), stochastic gradient descent algorithm is used with mini-batch size of 128. Applying Xavier initialization for the weights gives the filexibility to choose larger learning rates ([X. Glorot and Y. Bengio, 2010](http://proceedings.mlr.press/v9/glorot10a.html)). Thus, applying Xavier initialization, the learning rate is taken as 0.1 with an exponential decay. An early stopping algorithm is implemented within a limit of 200 epochs. After the training, the test set is evaluated with the best model weights performed on the validation set.

For LeNet-5, best validation accuracy is 92.58% and the test accuracy with the best model is 91.18%. The calculated f1-scores for each class can be seen below ([source](./Traffic_Sign_Classifier.ipynb)).

![lenet-f1-scores](./images/lenet-f1-scores.png)

F1-scores are given for each class. F1-score is a measure of precision and recall are used together. If the model predicts well on a class, that class should have high precision and recall values, which, here in this case, represented by the f1-score values very close to 1. As can be seen from the figure above, most of the classes have very poor f1-score values. For this reason, model improvements, and even a different model with data augmentation will be implemented.


### B) The LeNet Architecture with Dropout

Dropout technique prevents neural networks from overfitting ([N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever and R. Salakhutdinov, 2014](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)). The dropout refers to temporarily removing some of the randomly chosen hidden/visible units in a neural network on the training process. This way the dropout technique breaks up the co-adaptations of the nodes in a layer and the model generalizes better. Thus for the same LeNet-5 architecture above, dropout is implemented for the fully connected layers to generalize the model.

#### The results for LeNet-5 with dropout

The same process of traning is again used for the same LeNet-5 architecture by additionally implementing dropouts to fully connected layers.

For LeNet-5 with dropout, the best validation accuracy is 98.00% and the test accuracy with the best model is 95.73%. The calculated f1-scores for each class can be seen below ([source](./Traffic_Sign_Classifier.ipynb)).

![lenet-f1-scores](./images/lenet-dropout-f1-scores.png)

The accuracy and the f1-scores for each class are improved indicating that the model generalizes better when dropout is applied. However, for some of the classes, the model still predicts very poorly.


### C) The Alexnet Architecture

The AlexNet model is the [ILSVRC](http://www.image-net.org/challenges/LSVRC/)-2012 competition winning, deep convolutional neural network architecture which had error rates considerably better than the previous state-of-the-art results ([A. Krizhevsky, I. Sutskever and G. E. Hinton, 2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)). The major difference from the LeNet-5 architecture is that AlexNet has more convolutional layers as being a deeper model. The LeNet-5 was built to train a dataset with 10 clases, however, AlexNet was built to predict 1000 clases. In 2012, recently developed advanced GPUs had let this kind of extended deep convolutional network architectures to be trained in a reasonable time for large images of 1000 clases. The achievements of the AlexNet on large scale images revealed the significance of deep convolutional neural networks.

In the following years, it has also been shown by the other ILSVRC deep convolutional network models that the deeper models achieve better results ([K. Simonyan and A. Zisserman, 2014](https://arxiv.org/abs/1409.1556)). And even the skip connections help on very deep models that achieve even better results ([K. He, X. Zhang, S. Ren and J. Sun, 2015](https://arxiv.org/abs/1512.03385)). Since the dataset in this project has relatively small images and less classes (43 classes), a scaled-down AlexNet architecture ([scaled AlexNet](./scaledalexnet.py)) is used for the project. The scaled AlexNet has half the number of feature maps in each convolutional layer and half the number of nodes in fully connected layers. In addition, since the image size is relatively small, the kernel sizes are choosen smaller than it is in the original model. And some of the operations are not applied such as Local Response Normalisation which has been reported that it does not improve the model performance ([K. Simonyan and A. Zisserman, 2014](https://arxiv.org/abs/1409.1556)).

#### Details of the scaled AlexNet architecture

The input images have 32x32x3 dimensions. Layer 1 is a convoltional layer with 3x3 kernel size and 1x1 strides which is followed with ReLu activation operation and 2x2 max pooling with a stride of 2x2. Layer 1 outputs 48 feature maps. Layer 2 has the same kernel as 3x3 with 1x1 stride and a ReLu operation is applied at the end. Layer 2 outputs 128 feature maps. Layer 3 has also 3x3 kernel size and 1x1 stride followed by a ReLu operation, outputs 192 feature maps. Layer 4 is also a convolutional layer with 2x2 kernel size, 1x1 stride and again ReLu operation is applied that outputs 192 feature maps. Layer 5 is the last convolutinal layer having 3x3 kernel size and 1x1 stride. A maxpooling of 2x2 is applied with 2x2 strides after the ReLu operation. The number of the feature maps that layer 5 outputs is 128. After layer 5, the output nodes are flatten forming the layer 6 and fully connected to the following layer with 1024 output connections. Layer 7 is also a fully connected layer with 1024 input and 1024 output connections. Dropout technique after the ReLu operations is applied for both of layer 6 and layer 7. Layer 8 is the output layer with 43 outputs nodes as being the class logits ([source](./scaledalexnet.py)).


#### Data augmentation

To avoid overfitting, for the scaled AlexNet model, data augmentation is aplied on train and validation sets. In each set, each class with the number of examples are less than two thousand, rised up to a number of two thousand and two thousand and ten examples by appying random rotations, translations and viewpoint transformations to existing examples. By having two thousand and ten examples for the train and two thousand examples for the validation sets, the model fine tuned and trained to evaluate on test sets ([source](./data-augmentation-equal-numbers.ipynb)).

#### The results of scaled AlexNet

For the scaled AlexNet, the same [training process](./salexnet-equal-n/trainer.py) is used as it was for the LeNet-5 architecture, again with early stopping and limiting the maximum number of epochs to 200. The scaled AlexNet perform better than the LeNet-5 model as expected.

The scaled AlexNet achieved a test accuracy of 98.18% with the best model. The f1-scores for each class is plotted below ([source](./salexnet-jittered-datasets-equal-n.nbconvert.ipynb)).

![alexnet-f1-scores](./images/alexnet-f1-scores.png)

As can be seen from the plot of f1-scores, scaled AlexNet predicts better for all the classes and most of the classes have a f1-score value higher than 0.85.

#### Ensemble averaging

It has been reported by the winner of the [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb) competition that they used multiple convolutional neural network models to average out their performance ([D. Ciresan, U. Meier, J. Masci and J. Schmidhuber, 2012](http://www.sciencedirect.com/science/article/pii/S0893608012000524)). For the same purpose, ten identical scaled AlexNet models initialized with Xavier weights and the logits of the three models with the best performance are averaged to get a higher accuracy.

| Model No              |  Validation Accuracy       |   Test Accuracy       |   Number of epochs    |
|:---------------------:|:--------------------------:|:---------------------:|:---------------------:| 
| 1                     |  96.66%                    |  97.69%               |  200                  |
| 2                     |  95.18%                    |  97.66%               |  25                   |
| **3**                 |  **95.56%**                |  **98.03%**           |  **51**               |
| **4**                 |  **96.03%**                |  **98.19%**           |  **55**               |
| 5                     |  96.28%                    |  97.76%               |  57                   |
| 6                     |  95.93%                    |  97.88%               |  21                   |
| **7**                 |  **96.56%**                |  **98.00%**           |  **65**               |
| 8                     |  96.45%                    |  97.66%               |  79                   |
| 9                     |  94.91%                    |  97.47%               |  27                   |
| 10                    |  95.06%                    |  97.45%               |  21                   |


The three models with the best test accuracy (model 3, 4 and 7) are used for ensemble averaging. The f1-scores are given below for these three models.

![ensemble-f1-scores](./images/ens-f1-scores.png)

As can be seen from the plot of f1-scores, each model has a different performance for each class. Thus and averaging of the logits of these models can improve the accuracy on the test set evaluation.


By averaging the logits of model 3,4 and 7, accuracy on the test set becomes 98.38% ([source](./ensemble-average-evaluation.ipynb)). The reason each trained model has a different epoch number is, during the training, an early stopping algorithm was used.


### Improving The Performance

The winning team of the [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb) competition achived 99.46% accuracy on the test set ([D. Ciresan, U. Meier, J. Masci and J. Schmidhuber, 2012](http://www.sciencedirect.com/science/article/pii/S0893608012000524)). And the average human accuracy is reported as 98.84 ([J. Stallkamp, M. Schlipsing, J. Salmen and C. Igel, 2012](http://www.sciencedirect.com/science/article/pii/S0893608012000457)).

With ensemble averaging, scaled AlexNet got an accuracy of 98.38% on the test set. To get even higher accuracies, it is possible to apply extra data augmentation techniques. For instance, since the examples have different lighting condisions, applying histogram equalization could help the model to generalize better. In addition, no skip connections ([K. He, X. Zhang, S. Ren and J. Sun, 2015](https://arxiv.org/abs/1512.03385)) or deeper models ([C. Szegedy et al., 2014](https://arxiv.org/abs/1409.4842)) are used in the project, while skip connections are reported as an major factor to improve the performance for other models ([P. Sermanet and Y. LeCun, 2011](http://ieeexplore.ieee.org/document/6033589/)). In addition, using the same model, without implementing an early stopping algorithm and training the model to even higher number of epochs could help stochastic gradient descent algorithm to find a better minimum for its loss function to improve the performance.


### Testing on New Examples

Five additional German traffic examples are captured from a video stream on the net and evaluated the previously trained scaled AlexNet on these new examples.

All the images are captured in a rainy weather to be able to test the model on hard examples. These five images are listed below.

|    Images                                                      |  Name                 |  size             |
|:--------------------------------------------------------------:|:---------------------:|:-----------------:|
| ![No passing](./collected-data/resized/resized_9.png)          |  No passing           |  32x32x3          |
| ![Priority road](./collected-data/resized/resized_12.png)      |  Priority road        |  32x32x3          |
| ![Dangerous curve right](./collected-data/resized/resized_20.png) | Dangerous curve to the right | 32x32x3 |
| ![Speed limit 80](./collected-data/resized/resized_5.png)      |  Speed limit (80km/h) |  32x32x3          |
| ![Speed limit 60](./collected-data/resized/resized_3.png)      |  Speed limit (60km/h) |  32x32x3          |


The scaled AlexNet model evaluated on these images by initializing the weights of the best model with the test accuracy of 98.18%. As the result, the model got an accuracy of 80% and predicted 4/5 examples correctly. Top 5 softmax probabilities are given for the each image below.

#### Top5 probability for image 1 (No passing)

![No passing](./collected-data/class_9.png) 

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| **1.00000000e+00**    | **No passing**                                | 
| 1.35289637e-08        | No passing for vehicles over 3.5 metric tons  |
| 1.06118925e-09        | Speed limit (60km/h)                          |
| 1.15808446e-10        | Speed limit (50km/h)                          |
| 9.19384846e-11        | Speed limit (120km/h)                         |


#### Top5 probability for image 2 (Priority road)

![Priority road](./collected-data/class_12.png) 

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| **1.00000000e+00**    | **Priority road**                             | 
| 3.09655501e-09        | Right-of-way at the next intersection         |
| 1.59932126e-12        | No entry                                      |
| 7.51019310e-13        | Traffic signals                               |
| 3.61374044e-13        | End of no passing by vehicles over 3.5 metric tons |


#### Top5 probability for image 3 (Dangerous curve to the right)

![Dangerous curve right](./collected-data/class_20.png) 

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| **1.00000000e+00**    | **Dangerous curve to the right**              | 
| 6.29055308e-10        | Dangerous curve to the left                   |
| 2.16328122e-10        | Slippery road                                 |
| 3.47970784e-11        | No passing                                    |
| 8.59441099e-14        | Vehicles over 3.5 metric tons prohibited      |


#### Top5 probability for image 4 (Speed limit (80km/h))

![Speed limit 80](./collected-data/class_5.png)

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| **9.99999285e-01**    | **Speed limit (80km/h)**                      | 
| 7.26892040e-07        | Speed limit (60km/h)                          |
| 3.31280847e-09        | Speed limit (50km/h)                          |
| 4.42142919e-11        | Speed limit (30km/h)                          |
| 1.05825152e-13        | No passing for vehicles over 3.5 metric tons  |


#### Top5 probability for image 5 (Speed limit (60km/h))

![No passing](./collected-data/class_3.png) 

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 9.99993682e-01        | Speed limit (20km/h)                          | 
| 6.11884252e-06        | Speed limit (120km/h)                         |
| 1.69211575e-07        | Children crossing                             |
| **1.53469841e-07**    | **Speed limit (60km/h)**                      |
| 2.34251694e-08        | No passing                                    |