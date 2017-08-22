import os
import tensorflow as tf
import timeit
from functools import reduce
import numpy as np
from six.moves import cPickle as pickle

# The Trainer class

class Trainer(object):
    """A class to train a given model.

    This class trains a model with a early stopping feature. Saves the tensorflow session
    with the best weight parameters for the model with the highest validation accuracy 
    during the training. After the training, prints the last and also the best testset 
    accuracies.

    Requires `timeit`, `TensorFlow`, `reduce` from `functools`, `os` and `numpy` libraries.

    Attributes:
        datasets: A 3x2 tuple with the elements of numpy arrays as the datasets of train,
                  validation and test features and labels. An example is: 
                  ((X_train, y_train), (X_validation, y_validation), (X_test, y_test))
        n_epochs: An integer to represent the training run time limit with the max number of 
                  epochs to run.
    """

    def __init__(self, datasets, batch_size=128, n_epochs=200):

        self.X_train, self.y_train = datasets[0]
        self.X_valid, self.y_valid = datasets[1]
        self.X_test, self.y_test = datasets[2]
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def randomize(self, dataset, labels):
        """Randomize a given dataset.
        
        Args:
            dataset: A numpy array to represent the features (X) of a dataset.
            labels: A numpy array tor represent the labels (y) of a dataset.
        """
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation,:,:]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels

    def run(self, model, model_save_dir, n_epochs=None):
        """Start training.

        Call this function to start the training.
        
        Attributes:
            model: A python class instance defining the classifier.
            model_save_dir: A string to represent the name of the directory 
                            that pickled f1-scores and (best) tensorflow 
                            session will be saved.
            n_epochs: An integer to represent the training run time limit with the max number of 
                      epochs to run. If the value is `None` then `self.epoch` will be used.
        """
        
        if model.batch_size != self.batch_size:
            raise ValueError('Model batch size and the trainer batch size is not equal.')
            
        # mkdir
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            
        if n_epochs is None:
            n_epochs = self.n_epochs
        
        # Init the model
        model.init_model()
            
        X_train, y_train = self.X_train, self.y_train
        X_valid, y_valid = self.X_valid, self.y_valid
        X_test, y_test = self.X_test, self.y_test
        
        self.saver = tf.train.Saver()

        # early-stopping parameters
        patience = 5000 # look as this many examples regardless
        patience_increase = 2
        improvement_threshold = 0.995 # a relative improvement of 
                                          # this much is considered significant
        best_valid_loss = np.inf
        start_time = timeit.default_timer()
        prev_time = start_time
        
        n_train_batches = X_train.shape[0] // self.batch_size
        valid_freq = min(n_train_batches, patience // 2)
        epoch_freq_ratio = max(1, n_train_batches // (patience // 2))
        
        config = tf.ConfigProto(allow_soft_placement = True)
        
        delta_t = []
        with tf.Session(config=config) as sess:
            # Init vars
            tf.global_variables_initializer().run()
            print('Initialized')
            done_looping = False
            epoch = 0
            while (epoch < n_epochs) and (not done_looping):
                epoch = epoch + 1
                for minibatch_index in range(n_train_batches):
                    
                    batch_X = \
                        X_train[minibatch_index * self.batch_size:(minibatch_index + 1) * self.batch_size, :]
                    batch_Y = \
                        y_train[minibatch_index * self.batch_size:(minibatch_index + 1) * self.batch_size]
                    feed_dict = {model.X: batch_X, model.Y: batch_Y}
                    _, l, train_accuracy = sess.run(
                                [model.optimizer, model.loss, 
                                model.train_accuracy_operation], feed_dict=feed_dict)
                
                    iter = (epoch - 1) * n_train_batches + minibatch_index # cumulative iteration number
                
                    if (iter + 1) % valid_freq == 0:
                        valid_accuracy, f1_scores = model.evaluate(X_valid, y_valid, sess=sess, f1score=True)
                        this_valid_loss = 1. - valid_accuracy
                    
                        print("Minibatch loss at epoch %i and iter %i: %f and the learning rate: %f" % 
                              (epoch, iter, l, model.learning_rate.eval()))
                        print("Minibatch train and validation accuracy: %.3f%%, %.3f%%" 
                            % (train_accuracy * 100., valid_accuracy * 100.))
                        
                        end_time = timeit.default_timer()
                        delta_t.append(end_time - prev_time)
                        print('Time interval: %.4f seconds, estimated run time for %i epochs: %.4f hours' % 
                                ((end_time - prev_time), n_epochs, 
                                ((reduce(lambda x, y: x + y, delta_t) / len(delta_t)) 
                                 * n_epochs * epoch_freq_ratio) / 3600.))
                        #print(f1_scores)
                        prev_time = end_time
                    
                        if this_valid_loss < best_valid_loss:
                            if this_valid_loss < best_valid_loss * improvement_threshold:
                                patience = max(patience, iter * patience_increase)
                            
                            best_valid_loss = this_valid_loss
                            
                            # save for the best model
                            self.saver.save(sess, model_save_dir + '/best-model-session')
                            # save f1-scores for the best valid score
                            with open(model_save_dir + '/best-valid-f1-scores.pkl', 'wb') as f:
                                pickle.dump(f1_scores, f)
                            print('Model saved')
                
                    if patience <= iter:
                            done_looping = True
                            break
            
            test_accuracy, f1_scores = model.evaluate(X_test, y_test, sess=sess, f1score=True)
            print("Final Test accuracy: %.1f%%" 
                          % (test_accuracy * 100.))
            #print(f1_scores)
            sess.close()
        
        
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, model_save_dir + '/best-model-session')
            
            test_accuracy, f1_scores = model.evaluate(X_test, y_test, sess=sess, f1score=True)
            print("Test accuracy with the best model: %.3f%%" 
                          % (test_accuracy * 100.))
            print('f1-scores of classes:')
            print(f1_scores)
            with open(model_save_dir + '/best-test-f1-scores.pkl', 'wb') as f:
                pickle.dump(f1_scores, f)
            sess.close()
            
        
        end_time = timeit.default_timer()
        
        print('Total run time %.4f minutes' % ((end_time - start_time) / 60.))
