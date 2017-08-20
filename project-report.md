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

