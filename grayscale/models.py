import tensorflow as tf
import numpy as np

# Model Classes

# The class of a scaled AlexNet (sAlexNet)
class ScaledAlexNet(object):
    """A class defining a scaled AlexNet model with the TensorFlow framework.

    Attributes:
        num_labels: An integer representing the number of classes.
        image_size: An integer representing the length of one side of a suqared image in pixels.
        learning_rate: A float to represent the learning rate of the optimization process.
        batch_size: An integer to represents the number of examples in each batch.
        decay_interval: An integer to represent decay interval.

    """

    def __init__(self, num_labels, image_size=32, learning_rate=0.01, batch_size=128,
                 decay_interval=5000):

        self.num_labels = num_labels
        self.alpha = learning_rate
        self.batch_size = batch_size
        self.image_size = image_size
        self.decay_interval = decay_interval
        
    def init_model(self):
        """Call this method just before using the model."""
        
        tf.reset_default_graph()
        
        # Model Variables
        self.W_1 = tf.get_variable("W1", shape=[3, 3, 1, 48],
                                       initializer=tf.contrib.layers.xavier_initializer())
        self.b_1 = tf.Variable(tf.zeros([48]))
        self.W_2 = tf.get_variable("W2", shape=[3, 3, 48, 128],
                                       initializer=tf.contrib.layers.xavier_initializer())
        self.b_2 = tf.Variable(tf.zeros([128]))
        self.W_3 = tf.get_variable("W3", shape=[3, 3, 128, 192],
                                       initializer=tf.contrib.layers.xavier_initializer())
        self.b_3 = tf.Variable(tf.zeros([192]))
        self.W_4 = tf.get_variable("W4", shape=[2, 2, 192, 192],
                                       initializer=tf.contrib.layers.xavier_initializer())
        self.b_4 = tf.Variable(tf.zeros([192]))
        self.W_5 = tf.get_variable("W5", shape=[3, 3, 192, 128],
                                       initializer=tf.contrib.layers.xavier_initializer())
        self.b_5 = tf.Variable(tf.zeros([128]))
        
        self.W_6 = tf.get_variable("W6", shape=[4 * 4 * 128, 1024],
                                       initializer=tf.contrib.layers.xavier_initializer())
        self.b_6 = tf.Variable(tf.zeros([1024]))
        self.W_7 = tf.get_variable("W7", shape=[1024, 1024],
                                       initializer=tf.contrib.layers.xavier_initializer())
        self.b_7 = tf.Variable(tf.zeros([1024]))
        self.W_8 = tf.get_variable("W8", shape=[1024, self.num_labels],
                                       initializer=tf.contrib.layers.xavier_initializer())
        self.b_8 = tf.Variable(tf.zeros([self.num_labels]))

        # Parameters
        self.global_step = tf.Variable(0)
        self.learning_rate = tf.train.exponential_decay(self.alpha, self.global_step, 
                                                        self.decay_interval, 
                                                        0.95, staircase=True)
        self.X = tf.placeholder(
                    tf.float32, shape=(self.batch_size, self.image_size, self.image_size, 1))
        self.Y = tf.placeholder(tf.int32, shape=(self.batch_size))
        Y_one_hot = tf.one_hot(self.Y, self.num_labels)

        # Get logits
        train_logits, dropout_logits = self.get_logits(self.X, dropout=True)
        self.logits = self.get_logits(self.X, dropout=False)
        # Get loss
        self.loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y_one_hot, logits=dropout_logits))
        # Optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)\
                            .minimize(self.loss, global_step=self.global_step)
        # Prediction and accuracy for train dataset
        train_prediction = tf.equal(tf.argmax(train_logits, 1), tf.argmax(Y_one_hot, 1))
        self.train_accuracy_operation = tf.reduce_mean(tf.cast(train_prediction, tf.float32))
        # Prediction and accuracy for validation and test datasets
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(Y_one_hot, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # F1 scoring
        predictions = tf.one_hot(tf.argmax(self.logits, 1), self.num_labels, dtype=tf.int32)
        actuals = tf.to_int32(Y_one_hot)
        self.TP = tf.count_nonzero(predictions * actuals, axis=0)
        self.FP = tf.count_nonzero(predictions * (actuals - 1), axis=0)
        self.FN = tf.count_nonzero((predictions - 1) * actuals, axis=0)



    def get_logits(self, X, dropout=False):
        """Get the logits of the top layer.
        
        If `dropout` is `True`, logits are returned for both dropout 
        applied and not. If `False`, only the logits returned that 
        dropout is not applied.
        """
        
        conv = tf.nn.conv2d(X, self.W_1, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.relu(conv + self.b_1)
        pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        conv = tf.nn.conv2d(pool, self.W_2, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.relu(conv + self.b_2)
        conv = tf.nn.conv2d(hidden, self.W_3, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.relu(conv + self.b_3)
        conv = tf.nn.conv2d(hidden, self.W_4, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.relu(conv + self.b_4)
        conv = tf.nn.conv2d(hidden, self.W_5, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.relu(conv + self.b_5)
        pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        
        shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden1 = tf.nn.relu(tf.matmul(reshape, self.W_6) + self.b_6)
        def apply_dropout(hidden):
            dropout = tf.nn.dropout(hidden, 0.5)
            hidden = tf.nn.relu(tf.matmul(dropout, self.W_7) + self.b_7)
            dropout = tf.nn.dropout(hidden, 0.5)
            logits = tf.matmul(dropout, self.W_8) + self.b_8
            return logits
        hidden2 = tf.nn.relu(tf.matmul(hidden1, self.W_7) + self.b_7)
        logits = tf.matmul(hidden2, self.W_8) + self.b_8
        if dropout:
            return logits, apply_dropout(hidden1)
        else:
            return logits


    def evaluate(self, data_X, data_Y, sess=None, f1score=False):
        """Get accuracy using minibatches.
        
        Use it for validation and test datasets.
        """

        if sess is None:
            sess = tf.get_default_session()
        n_batches = data_X.shape[0] // self.batch_size
        total_accuracy = 0.
        if f1score:
            total_TP = np.zeros(data_Y.max() + 1, dtype=np.float32)
            total_FP = np.zeros(data_Y.max() + 1, dtype=np.float32)
            total_FN = np.zeros(data_Y.max() + 1, dtype=np.float32)
        for minibatch_index in range(n_batches):
            batch_X = data_X[minibatch_index * self.batch_size:(minibatch_index + 1) * self.batch_size]
            batch_Y = data_Y[minibatch_index * self.batch_size:(minibatch_index + 1) * self.batch_size]
            if f1score:
                accuracy, TP_, FP_, FN_ = sess.run([self.accuracy_operation,
                                            self.TP, self.FP, self.FN],
                                            feed_dict={self.X: batch_X, self.Y: batch_Y})
                total_TP += TP_
                total_FP += FP_
                total_FN += FN_
            else:
                accuracy = sess.run([self.accuracy_operation],
                                        feed_dict={self.X: batch_X, self.Y: batch_Y})
            total_accuracy += (accuracy * self.batch_size)
        accuracy = total_accuracy / (n_batches * self.batch_size)
        if f1score:
            precision = total_TP / (total_TP + total_FP + 1e-6)
            recall = total_TP / (total_TP + total_FN + 1e-6)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
            return accuracy, f1_scores
        else:
            return accuracy


    def get_class_scores(self, data_X, sess):
        """Get class scores using minibatches.
        
        Returns the score for each class.
        """
        n_batches = data_X.shape[0] // self.batch_size
        for minibatch_index in range(n_batches):
            batch_X = data_X[minibatch_index * self.batch_size:(minibatch_index + 1) * self.batch_size]
            logits_ = sess.run([self.logits],
                                feed_dict={self.X: batch_X})
            if minibatch_index == 0:
                scores = np.copy(logits_)
            else:
                scores = np.concatenate((scores, logits_), axis=0)
        return scores
        
        
