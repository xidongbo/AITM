'''
2022-08-16
Tensorflow implementation of Adaptive Information Transfer Multi-task (AITM) framework.
The source code for the paper: https://arxiv.org/abs/2105.08489.
@author: xidongbo
python = 3.6
tensorflow = 1.10.0
'''
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import argparse
import random
from sklearn.metrics import roc_auc_score
import multiprocessing
import queue
import threading
import shutil
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

def print_info(prefix, result, time):
    line = ['Task{}: AUC:%.6f'.format(task_idx) for task_idx in range(len(result))]
    print(prefix + ('[%.1fs]: \n' + '\n'.join(line)) % tuple([time] + result))


class GeneratorEnqueuer(object):
    """From keras source code training.py
    Builds a queue out of a data generator.

    # Arguments
        generator: a generator function which endlessly yields data
        pickle_safe: use multiprocessing if True, otherwise threading
    """

    def __init__(self, generator, pickle_safe=False):
        self._generator = generator
        self._pickle_safe = pickle_safe
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.finish = False

    def start(self, workers=1, max_q_size=10, wait_time=0.05):
        """Kicks off threads which add data from the generator into the queue.

        # Arguments
            workers: number of worker threads
            max_q_size: queue size (when full, threads could block on put())
            wait_time: time to sleep in-between calls to put()
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._pickle_safe or self.queue.qsize() < max_q_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except StopIteration:
                    self.finish = True
                    break
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            if self._pickle_safe:
                self.queue = multiprocessing.Queue(maxsize=max_q_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._pickle_safe:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed()
                    thread = multiprocessing.Process(
                        target=data_generator_task)
                    thread.daemon = True
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except BaseException:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stop running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called start().

        # Arguments
            timeout: maximum time to wait on thread.join()
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._pickle_safe:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._pickle_safe:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None


class AITM(object):
    def __init__(self, data_path=None, epoch=10, batch_size=2000, embedding_dim=5, layers=[128,64,32], keep_prob=[0.9,0.7,0.7], batch_norm=1, lamda=1e-6, lr=1e-3, 
                 optimizer='adam', verbose=1, activation='relu', early_stop=1, model_path='AITM', loss_weight=[1.,1.,1.,1.], num_tasks=4, 
                 constraint_weight=0.3, random_seed=2022):
        # bind params to class
        try:
            vocabulary_df = pd.read_csv('config.csv')
        except:
            print('The file "config.csv" (two columns: feature_name, vocabulary_size_of_the_feature) is needed for get the vocabulary size of all features.')
            exit()
        self.vocabulary_size = dict(zip(vocabulary_df.iloc[:, 0].to_list(), vocabulary_df.iloc[:, 1].to_list()))
        assert data_path is not None, 'The data_path(train/dev/test all is ok) is needed for getting the order of all features.'
        with open(data_path) as fr:
            self.all_features = fr.readline().strip().split(',')[num_tasks:]
        self.epoch = epoch
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.keep_prob = np.array(keep_prob)
        self.no_dropout = np.array([1 for _ in range(len(keep_prob))])
        self.batch_norm = batch_norm
        self.lamda = lamda
        self.lr = lr
        self.optimizer = optimizer
        self.verbose = verbose
        self.activation = activation
        self.early_stop = early_stop
        self.model_path = model_path
        self.loss_weight = loss_weight
        self.num_tasks = num_tasks
        self.constraint_weight = constraint_weight
        self.random_seed = random_seed
        self.trained = False
        self.loaded = False
        assert len(self.layers) == len(self.keep_prob), 'The length of "layers" and "keep_prob" should be same.'
        assert len(self.loss_weight) == self.num_tasks, 'The length of "loss_weight" should be same with the "num_tasks"'
        print('Params:' +
              '\n epoch:{}'.format(self.epoch) +
              '\n batch_size:{}'.format(self.batch_size) +
              '\n embedding_dim:{}'.format(self.embedding_dim) +
              '\n layers:{}'.format(self.layers) +
              '\n keep_prob:{}'.format(self.keep_prob) +
              '\n batch_norm:{}'.format(self.batch_norm) +
              '\n lamda:{}'.format(self.lamda) +
              '\n lr:{}'.format(self.lr) +
              '\n optimizer:{}'.format(self.optimizer) +
              '\n verbose:{}'.format(self.verbose) +
              '\n activation:{}'.format(self.activation) +
              '\n early_stop:{}'.format(self.early_stop) +
              '\n model_path:{}'.format(self.model_path) +
              '\n loss_weight:{}'.format(self.loss_weight) +
              '\n num_tasks:{}'.format(self.num_tasks) +
              '\n constraint_weight:{}'.format(self.constraint_weight) +
              '\n random_seed:{}'.format(self.random_seed)
              )

        # init all variables in a tensorflow graph
        self._init_graph_AITM()

    def _init_graph_AITM(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        print("Init raw AITM graph")
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Variables init.
            self.weights = self._initialize_weights()
            self.train_labels_placeholder = []
            for task_idx in range(self.num_tasks):
                self.train_labels_placeholder.append(tf.placeholder(tf.float64, shape=[None, 1], name='labels{}'.format(task_idx)))
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')
            self.dropout_keeps_placeholder = []
            for drop_idx in range(len(self.keep_prob)):
                self.dropout_keeps_placeholder.append(tf.placeholder(tf.float32, name='dropout_keep{}'.format(drop_idx)))

            # inputs
            self.inputs_placeholder = []
            for column in self.all_features:
                self.inputs_placeholder.append(tf.placeholder(
                    tf.int64, shape=[None, 1], name=column))

            feature_embedding = []
            for column, feature in zip(self.all_features, self.inputs_placeholder):
                embedded = tf.nn.embedding_lookup(self.weights['feature_embeddings_{}'.format(
                    column)], feature)  # [None, 1, K] * num_features
                feature_embedding.append(embedded)
            feature_embedding = tf.keras.layers.concatenate(feature_embedding, axis=-1)  # [None, 1, K * num_features]
            feature_embedding = tf.squeeze(feature_embedding, axis=1)  # [None, K * num_features]
            # AIT
            # for the first task, the ait = the tower
            self.tasks_outputs = []
            ait = tower = self.tower(feature_embedding, task_idx=0)
            task0_output = tf.keras.layers.Dense(1)(ait)
            self.tasks_outputs.append(tf.sigmoid(task0_output, name="pred0"))

            for task_idx in range(1, self.num_tasks):
                transfer = self.transfer(ait, task_idx=task_idx)
                tower = self.tower(feature_embedding, task_idx=task_idx)
                ait = self.attention(tower, transfer)
                taski_output = tf.keras.layers.Dense(1)(ait)
                self.tasks_outputs.append(tf.sigmoid(taski_output, name="pred{}".format(task_idx)))

            # for save model
            keys = [c + ':0' for c in self.all_features]
            self.inputs = dict(zip(keys, self.inputs_placeholder))
            self.inputs['train_phase:0'] = self.train_phase
            dropout_dict = dict(zip(['dropout_keep{}:0'.format(drop_idx) for drop_idx in range(len(self.keep_prob))], self.dropout_keeps_placeholder))
            self.inputs.update(dropout_dict)
            self.outputs = dict(zip(["pred{}:0".format(task_idx) for task_idx in range(self.num_tasks)], self.tasks_outputs))

            # Compute the loss.
            reg_variables = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)
            if self.lamda > 0:
                reg_loss = tf.add_n(reg_variables)
            else:
                reg_loss = 0.
            self.loss = reg_loss
            for task_idx in range(self.num_tasks):
                self.loss += self.loss_weight[task_idx] * tf.losses.log_loss(self.train_labels_placeholder[task_idx], self.tasks_outputs[task_idx])
            # -------label_constraint--------
            label_constraint = 0.
            for task_idx in range(1, self.num_tasks):
                label_constraint += tf.maximum(self.tasks_outputs[task_idx] - self.tasks_outputs[task_idx - 1], tf.zeros_like(self.tasks_outputs[0]))
            self.loss = self.loss + self.constraint_weight * tf.reduce_mean(label_constraint, axis=0)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # Optimizer.
                if self.optimizer == 'adam':
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9,
                                                            beta2=0.999, epsilon=1e-8).minimize(self.loss)
                elif self.optimizer == 'adagrad':
                    self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr,
                                                               initial_accumulator_value=1e-8).minimize(self.loss)
                elif self.optimizer == 'gd':
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(
                        self.loss)
                elif self.optimizer == 'moment':
                    self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr,
                                                                momentum=0.95).minimize(self.loss)

            # init
            self.saver = tf.train.Saver(var_list=tf.global_variables())
            init = tf.global_variables_initializer()
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.InteractiveSession(
                config=tf.ConfigProto(
                    gpu_options=gpu_options))
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            print("#params: %d" % total_parameters)

    def _initialize_weights(self):
        '''
        Init the parameters weights.
        :return: all weights.
        '''
        all_weights = dict()
        l2_reg = tf.contrib.layers.l2_regularizer(self.lamda)
        # attention
        all_weights['attention_w1'] = tf.get_variable(
            initializer=tf.random_normal(
                shape=[self.layers[-1], self.layers[-1]],
                mean=0.0,
                stddev=0.01),
            regularizer=l2_reg, name='attention_w1')  # k * k
        all_weights['attention_w2'] = tf.get_variable(
            initializer=tf.random_normal(
                shape=[self.layers[-1], self.layers[-1]],
                mean=0.0,
                stddev=0.01),
            regularizer=l2_reg, name='attention_w2')  # k * k
        all_weights['attention_w3'] = tf.get_variable(
            initializer=tf.random_normal(
                shape=[self.layers[-1], self.layers[-1]],
                mean=0.0,
                stddev=0.01),
            regularizer=l2_reg, name='attention_w3')  # k * k
        # MLP
        for column in self.all_features:
            all_weights['feature_embeddings_{}'.format(column)] = tf.get_variable(
                initializer=tf.random_normal(
                    shape=[
                        self.vocabulary_size[column],
                        self.embedding_dim],
                    mean=0.0,
                    stddev=0.01),
                regularizer=l2_reg, name='feature_embeddings_{}'.format(column))  # self.vocabulary_size * K
        return all_weights

    def attention(self, tower, transfer):
        '''
        compute the ait attention
        :param tower: the tower, i.e., the q_{t} in the paper, the shape is [None, K]
        :param transfer: the transferred info, i.e., the p_{t-1} in the paper, the shape is [None, K]
        :return: the ait output, the shape is [None, L, K]
        '''
        # (N,L,K)
        inputs = tf.concat([tower[:, None, :], transfer[:, None, :]], axis=1)
        # (N,L,K)*(K,K)->(N,L,K), L=2, K=32 in this.
        Q = tf.tensordot(inputs, self.weights['attention_w1'], axes=1)
        K = tf.tensordot(inputs, self.weights['attention_w2'], axes=1)
        V = tf.tensordot(inputs, self.weights['attention_w3'], axes=1)
        # (N,L)
        a = tf.reduce_sum(tf.multiply(Q, K), axis=-1) / \
            tf.sqrt(tf.cast(inputs.shape[-1], tf.float32))
        a = tf.nn.softmax(a, axis=1)
        # (N,L,K)
        outputs = tf.multiply(a[:, :, None], V)
        return tf.reduce_sum(outputs, axis=1)  # (N, K)

    def tower(self, feature_embedding, task_idx):
        '''
        compute the MLP tower, i.e., the q_{t} in the paper.
        :param task_idx: The task index
        :param feature_embedding: the embedded input.
        :return: the tower.
        '''
        for layer_idx in range(len(self.layers)):
            tower = tf.keras.layers.Dense(self.layers[layer_idx])(feature_embedding)
            if self.batch_norm:
                tower = self.batch_norm_layer(tower, self.train_phase, scope_bn='bn_{}_{}'.format(task_idx, layer_idx))
            tower = tf.keras.layers.Activation(activation=self.activation)(tower)
            tower = tf.nn.dropout(tower, self.dropout_keeps_placeholder[layer_idx])
        return tower

    def transfer(self, ait, task_idx):
        '''
        compute the transferred info, i.e., the p_{t-1} in the paper.
        :param task_idx: The task index.
        :param ait: the info of the last task, i.e., the z_{t-1} in the paper.
        :return: the transferred info.
        '''
        transfer = tf.keras.layers.Dense(self.layers[-1])(ait)
        if self.batch_norm:
            transfer = self.batch_norm_layer(transfer, self.train_phase, scope_bn='bn_{}_{}'.format(task_idx, len(self.layers)))
        transfer = tf.keras.layers.Activation(activation=self.activation)(transfer)
        transfer = tf.nn.dropout(transfer, self.dropout_keeps_placeholder[-1])
        return transfer

    def batch_norm_layer(self, x, train_phase, scope_bn):
        '''
        BN layer.
        :param x: the input.
        :param train_phase: is train phase?
        :param scope_bn: scope.
        :return: the output of BN layer.
        '''
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def feed(self, data, inputs=None, train_phase=True):
        '''
        fill the feed dict.
        :param inputs: In tf-serving mode, the inputs need be passed in. Other None.
        :param train_phase: is train_phase?
        :param data: x and y
        :return: the filled feed dict.
        '''
        if inputs is not None:
            # tf-serving
            feed_dict = {inputs[-1]: train_phase}
            # feed x
            for column_name, column_placeholder in zip(
                    self.all_features, inputs[:-len(self.keep_prob) - 1]):
                feed_dict[column_placeholder] = data['ids_{}'.format(column_name)]
            # feed dropout
            for layer_idx in range(len(self.layers)):
                feed_dict[inputs[ layer_idx - len(self.keep_prob) - 1]] = self.keep_prob[layer_idx] if train_phase else self.no_dropout[layer_idx]
        else:
            feed_dict = {self.train_phase: train_phase}
            # feed x
            for column_name, column_placeholder in zip(
                    self.all_features, self.inputs_placeholder):
                feed_dict[column_placeholder] = data['ids_{}'.format(column_name)]
            # feed y
            if inputs is None:
                for task_idx in range(self.num_tasks):
                    feed_dict[self.train_labels_placeholder[task_idx]] = data['Y{}'.format(task_idx)]
            # feed dropout
            for layer_idx in range(len(self.layers)):
                feed_dict[self.dropout_keeps_placeholder[layer_idx]] = self.keep_prob[layer_idx] if train_phase else self.no_dropout[layer_idx]
        return feed_dict

    def fit_on_batch(self, data):
        '''
        Fit on a batch data.
        :param data: a batch data.
        :return: The LogLoss.
        '''
        feed_dict = self.feed(data=data, train_phase=True)
        loss, _ = self.sess.run(
            (self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def fit(self, train_path, dev_path, max_q_size=20):
        '''
        Fit the train data.
        :param train_path: train path.
        :param dev_path:  validation path.
        :param max_q_size: maximum size for the generator queue.
        :return: None
        '''
        max_acc = -np.inf
        best_epoch = 0
        earlystop_count = 0
        enqueuer = None
        wait_time = 0.001  # in seconds
        for epoch in range(self.epoch):
            try:
                train_gen = self.iterator(train_path, shuffle=True)
                enqueuer = GeneratorEnqueuer(train_gen)
                enqueuer.start(max_q_size=max_q_size)
                t1 = time.time()
                train_loss = 0.
                nb_sample = 0
                i = 0
                while True:
                    # get a batch
                    generator_output = None
                    while enqueuer.is_running():
                        if not enqueuer.queue.empty():
                            generator_output = enqueuer.queue.get()
                            break
                        elif enqueuer.finish:
                            break
                        else:
                            time.sleep(wait_time)
                    # Fit training, return loss...
                    if generator_output is None:  # epoch end
                        break
                    nb_sample += len(generator_output['Y1'])
                    train_loss += self.fit_on_batch(generator_output)
                    if self.verbose > 0:
                        if (i + 1) % self.verbose == 0:
                            print('[%d]Train loss on step %d: %.6f' %
                                  (nb_sample, (i + 1), train_loss / (i + 1)))
                    i += 1
                # validation
                t2 = time.time()
                dev_gen = self.iterator(dev_path)
                true_pred = self.evaluate_generator(
                    dev_gen, max_q_size=max_q_size)
                valid_result = self.metrics(true_pred)

                print_info(
                    "Epoch %d [%.1f s]\t Dev" %
                    (epoch + 1, t2 - t1), valid_result, time.time() - t2)
                if self.early_stop > 0:
                    # 这里以所有任务在验证集上的AUC和最优为停止训练标志，如果只关注个别任务，可以设置为个别任务的AUC和
                    acc = sum(valid_result)
                    if max_acc >= acc:  # no gain
                        earlystop_count += 1
                    else:
                        self.save_path = self.saver.save(self.sess,
                                                         save_path='./{}.model'.format(self.model_path),
                                                         latest_filename='{}.check_point'.format(self.model_path))
                        max_acc = acc
                        best_epoch = epoch + 1
                        earlystop_count = 0
                    if earlystop_count >= self.early_stop:
                        print(
                            "Early stop at Epoch %d based on the best validation Epoch %d." % (
                                epoch + 1, best_epoch))
                        break

            finally:
                if enqueuer is not None:
                    enqueuer.stop()
        self.trained = True

    def evaluate_generator(self, generator, max_q_size=20,
                           workers=1, pickle_safe=False):
        '''
        See GeneratorEnqueuer Class about the following params.
        :param generator: the generator which return the data.
        :param max_q_size: maximum size for the generator queue
        :param workers: maximum number of processes to spin up
                when using process based threading
        :param pickle_safe: if True, use process based threading.
                Note that because
                this implementation relies on multiprocessing,
                you should not pass
                non picklable arguments to the generator
                as they can't be passed
                easily to children processes.
        :return: true labels, prediction probabilities.
        '''
        wait_time = 0.01
        enqueuer = None
        dev_y_trues = [[] for _ in range(self.num_tasks)]
        dev_y_preds = [[] for _ in range(self.num_tasks)]
        try:
            enqueuer = GeneratorEnqueuer(generator, pickle_safe=pickle_safe)
            enqueuer.start(workers=workers, max_q_size=max_q_size)
            nb_dev = 0
            while True:
                dev_batch = None
                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        dev_batch = enqueuer.queue.get()
                        break
                    elif enqueuer.finish:
                        break
                    else:
                        time.sleep(wait_time)
                if dev_batch is None:
                    break
                nb_dev += len(dev_batch['Y0'])
                feed_dict = self.feed(data=dev_batch, train_phase=False)
                predictions = self.sess.run(self.tasks_outputs, feed_dict=feed_dict)
                for task_idx in range(self.num_tasks):
                    dev_y_trues[task_idx] += list(dev_batch['Y{}'.format(task_idx)])
                    dev_y_preds[task_idx] += list(predictions[task_idx])

            # to row vectors
            for task_idx in range(self.num_tasks):
                dev_y_trues[task_idx] = np.reshape(dev_y_trues[task_idx], (-1,))
                dev_y_preds[task_idx] = np.reshape(dev_y_preds[task_idx], (-1,))
            print('Evaluate on %d samples.' % nb_dev)
        finally:
            if enqueuer is not None:
                enqueuer.stop()

        return dict(zip(['true{}'.format(task_idx) for task_idx in range(self.num_tasks)] +
                        ['pred{}'.format(task_idx) for task_idx in range(self.num_tasks)],
                        dev_y_trues + dev_y_preds))

    def evaluate_generator_serving(self, sess, generator, inputs, outputs, max_q_size=20,
                                   workers=1, pickle_safe=False):
        '''
        See GeneratorEnqueuer Class about the following params.
        :param generator: the generator which return the data.
        :param max_q_size: maximum size for the generator queue
        :param workers: maximum number of processes to spin up
                when using process based threading
        :param pickle_safe: if True, use process based threading.
                Note that because
                this implementation relies on multiprocessing,
                you should not pass
                non picklable arguments to the generator
                as they can't be passed
                easily to children processes.
        :return: true labels, prediction probabilities., event ids
        '''
        wait_time = 0.01
        enqueuer = None
        dev_y_trues = [[] for _ in range(self.num_tasks)]
        dev_y_preds = [[] for _ in range(self.num_tasks)]
        try:
            enqueuer = GeneratorEnqueuer(generator, pickle_safe=pickle_safe)
            enqueuer.start(workers=workers, max_q_size=max_q_size)
            nb_dev = 0
            while True:
                dev_batch = None
                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        dev_batch = enqueuer.queue.get()
                        break
                    elif enqueuer.finish:
                        break
                    else:
                        time.sleep(wait_time)
                # Fit training, return loss...
                if dev_batch is None:
                    break
                nb_dev += len(dev_batch['Y0'])
                feed_dict = self.feed(data=dev_batch, inputs=inputs, train_phase=False)
                predictions = sess.run(outputs, feed_dict=feed_dict)
                for task_idx in range(self.num_tasks):
                    dev_y_trues[task_idx] += list(dev_batch['Y{}'.format(task_idx)])
                    dev_y_preds[task_idx] += list(predictions[task_idx])
            # to row vectors
            for task_idx in range(self.num_tasks):
                dev_y_trues[task_idx] = np.reshape(dev_y_trues[task_idx], (-1,))
                dev_y_preds[task_idx] = np.reshape(dev_y_preds[task_idx], (-1,))
            print('Evaluate on %d samples.' % nb_dev)
        finally:
            if enqueuer is not None:
                enqueuer.stop()
        return dict(zip(['true{}'.format(task_idx) for task_idx in range(self.num_tasks)] +
                        ['pred{}'.format(task_idx) for task_idx in range(self.num_tasks)],
                        dev_y_trues + dev_y_preds))

    def iterator(self, lines, shuffle=False):
        '''
        Generator of data.
        Note that the csv in path should have been shuffle before here for train.
        :param shuffle: weather to shuffle the data.
        :param lines: list of all lines.
        :return: a batch data.
        '''
        prefetch = 50  # *batch_size
        batch_lines = []
        with open(lines, 'r') as fr:
            lines = []
            # remove csv header
            fr.readline()
            for prefetch_line in fr:
                lines.append(prefetch_line)
                if len(lines) >= self.batch_size * prefetch:
                    if shuffle:
                        random.shuffle(lines)
                    for line in lines:
                        batch_lines.append(line.split(','))
                        if len(batch_lines) >= self.batch_size:
                            batch_array = np.array(batch_lines)
                            batch_lines = []
                            batch_data = {}
                            feat_start_index = self.num_tasks
                            for task_idx in range(self.num_tasks):
                                batch_data['Y{}'.format(task_idx)] = batch_array[:, task_idx: task_idx + 1].astype(np.float64)
                            for i, column in enumerate(self.all_features):
                                batch_data['ids_{}'.format(
                                    column)] = batch_array[:, i + feat_start_index:i + feat_start_index + 1].astype(np.int64)
                            yield batch_data
                    lines = []
            if 0 < len(lines) < self.batch_size * prefetch:
                if shuffle:
                    random.shuffle(lines)
                for line in lines:
                    batch_lines.append(line.split(','))
                    if len(batch_lines) >= self.batch_size:
                        batch_array = np.array(batch_lines)
                        batch_lines = []
                        batch_data = {}
                        feat_start_index = self.num_tasks
                        for task_idx in range(self.num_tasks):
                            batch_data['Y{}'.format(task_idx)] = batch_array[:, task_idx: task_idx + 1].astype(
                                np.float64)
                        for i, column in enumerate(self.all_features):
                            batch_data['ids_{}'.format(
                                column)] = batch_array[:, i + feat_start_index:i + feat_start_index + 1].astype(
                                np.int64)
                        yield batch_data
                if 0 < len(batch_lines) < self.batch_size:
                    batch_array = np.array(batch_lines)
                    batch_data = {}
                    feat_start_index = self.num_tasks
                    for task_idx in range(self.num_tasks):
                        batch_data['Y{}'.format(task_idx)] = batch_array[:, task_idx: task_idx + 1].astype(np.float64)
                    for i, column in enumerate(self.all_features):
                        batch_data['ids_{}'.format(
                            column)] = batch_array[:, i + feat_start_index:i + feat_start_index + 1].astype(np.int64)
                    yield batch_data

    def metrics(self, true_pred):
        '''
        Evaluation Metrics.
        :param true_pred: input json.
        :return: auc_score
        '''
        aucs = []
        for task_idx in range(self.num_tasks):
            aucs.append(roc_auc_score(y_true=true_pred['true{}'.format(task_idx)], y_score=true_pred['pred{}'.format(task_idx)]))
        return aucs

    def save(self, model_path):
        '''
        save the tf-serving model
        :param model_path: the mode path.
        :return: the mode path.
        '''
        if not self.trained:
            print('Please first fit the model.')
            return
        # restore the best epoch model
        self.saver.restore(self.sess, save_path=self.save_path)

        # save as tf serving for online predict.
        serving_save_path = model_path
        with self.graph.as_default():
            if os.path.exists(serving_save_path):
                shutil.rmtree(serving_save_path)
            builder = tf.saved_model.builder.SavedModelBuilder(serving_save_path)
            inputs = {}
            for name in self.inputs:
                inputs[name] = tf.saved_model.utils.build_tensor_info(self.inputs[name])
            outputs = dict(zip(['pred{}:0'.format(task_idx) for task_idx in range(self.num_tasks)],
                               [tf.saved_model.utils.build_tensor_info(self.outputs['pred{}:0'.format(task_idx)]) for
                                task_idx in range(self.num_tasks)]))

            signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs,
                                                                               tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(self.sess, [tf.saved_model.tag_constants.SERVING],
                                                 signature_def_map={'test_signature': signature},
                                                 legacy_init_op=legacy_init_op)
            builder.save()
            print("tf-serving model is saved in: {}".format(model_path))
        return model_path

    def load(self, model_path):
        '''
        load the tf-serving model.
        :param model_path: the mode path.
        :return: the mode path
        '''
        # with tf.Session() as sess:
        sess = tf.Session()
        # load model
        meta_graph_def = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], model_path)
        # get signature
        signature = meta_graph_def.signature_def
        key_my_signature = 'test_signature'
        # get tensor name
        train_ids = []
        for column in self.all_features:
            train_id = signature[key_my_signature].inputs['{}:0'.format(column)].name
            train_ids.append(sess.graph.get_tensor_by_name(train_id))
        for drop_idx in range(len(self.keep_prob)):
            train_ids.append(sess.graph.get_tensor_by_name(
                signature[key_my_signature].inputs['dropout_keep{}:0'.format(drop_idx)].name))
        train_ids.append(sess.graph.get_tensor_by_name(signature[key_my_signature].inputs['train_phase:0'].name))
        preds = []
        for task_idx in range(self.num_tasks):
            preds.append(sess.graph.get_tensor_by_name(
                signature[key_my_signature].outputs['pred{}:0'.format(task_idx)].name))
        self.loaded = True
        self.sess = sess
        self.train_ids = train_ids
        self.preds = preds
        print('Loaded tf-serving model from {}'.format(model_path))
        return model_path

    def evaluate(self, dev_path):
        '''
        evaluate on the dev_path
        :param dev_path: data path to evaluate
        :return: the AUC list.
        '''
        if not self.trained and not self.loaded:
            print('Please first fit or load the model.')
            return
        if self.loaded:
            dev_gen = self.iterator(dev_path)
            true_pred = self.evaluate_generator_serving(self.sess, dev_gen, inputs=self.train_ids, outputs=self.preds)
            aucs = self.metrics(true_pred)
        elif self.trained:
            dev_gen = self.iterator(dev_path)
            true_pred = self.evaluate_generator(dev_gen)
            aucs = self.metrics(true_pred)
        return aucs

    def predict(self, test_path):
        '''
        predict on the test_path
        :param test_path: the data path
        :return: the predict probs dict for all tasks. {'pred0':[0.1,..],'pred1':[0.2,...]}
        '''
        if not self.trained and not self.loaded:
            print('Please first fit or load the model.')
            return
        if self.loaded:
            dev_gen = self.iterator(test_path)
            true_pred = self.evaluate_generator_serving(self.sess, dev_gen, inputs=self.train_ids, outputs=self.preds)
        elif self.trained:
            dev_gen = self.iterator(test_path)
            true_pred = self.evaluate_generator(dev_gen)
        preds = {}
        for task_idx in range(self.num_tasks):
            preds['pred{}'.format(task_idx)] = true_pred['pred{}'.format(task_idx)]
        return preds


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser(description="Run AITM.")
        parser.add_argument('--epoch', type=int, default=10,
                            help='Number of epochs.')
        parser.add_argument('--batch_size', type=int, default=2000,
                            help='Batch size.')
        parser.add_argument('--embedding_dim', type=int, default=5,
                            help='Number of embedding dim.')
        parser.add_argument('--layers', nargs='?', default='[128,64,32]',
                            help="Size of each MLP layer.")
        parser.add_argument('--keep_prob', nargs='?', default='[0.9,0.7,0.7]',
                            help='Keep probability. 1: no dropout.')
        parser.add_argument('--batch_norm', type=int, default=1,
                            help='Whether to perform batch normalization (0 or 1)')
        parser.add_argument('--lamda', type=float, default=1e-6,
                            help='Regularizer weight.')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='Learning rate.')
        parser.add_argument('--optimizer', nargs='?', default='adam',
                            help='Specify an optimizer type (adam, adagrad, gd, moment).')
        parser.add_argument('--verbose', type=int, default=1,
                            help='Whether to show the training process (0, or N batches show one time).')
        parser.add_argument('--activation', nargs='?', default='relu',
                            help='Which activation function to use for MLP layers (relu, sigmoid, tanh, identity).')
        parser.add_argument('--early_stop', type=int, default=1,
                            help='Whether to perform early stop (0, 1 ... any positive integer).')
        parser.add_argument('--model_path', nargs='?', default='AITM',
                            help='model_path for model_name path.')
        parser.add_argument('--gpu', nargs='?', default='0',
                            help='Which gpu to use.')
        parser.add_argument('--loss_weight', nargs='?', default='[1.,1.,1.,1.]',
                            help='loss weight for each task. The list size should be same as the num_tasks.')
        parser.add_argument('--num_tasks', type=int, default=4,
                            help='The number of tasks (>=2).')
        parser.add_argument('--constraint_weight', type=float, default=0.3,
                            help='label constraint weight.')
        parser.add_argument('--train_path', nargs='?', required=True,
                            help='The path of the train data.')
        parser.add_argument('--dev_path', nargs='?', required=True,
                            help='The path of the validation data.')
        parser.add_argument('--test_path', nargs='?', required=False,
                            help='The path of the test data.')
        parser.add_argument('--random_seed', type=int, default=2022,
                            help='Random seed.')
        return parser.parse_args()

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Training
    t1 = time.time()
    model = AITM(data_path=args.train_path,
                 epoch=args.epoch,
                 batch_size=args.batch_size,
                 embedding_dim=args.embedding_dim,
                 layers=eval(args.layers),
                 keep_prob=eval(args.keep_prob),
                 batch_norm=args.batch_norm,
                 lamda=args.lamda,
                 lr=args.lr,
                 optimizer=args.optimizer,
                 verbose=args.verbose,
                 activation=args.activation,
                 early_stop=args.early_stop,
                 model_path=args.model_path,
                 loss_weight=eval(args.loss_weight),
                 num_tasks=args.num_tasks,
                 constraint_weight=args.constraint_weight,
                 random_seed=args.random_seed)
    ## Training
    model.fit(args.train_path, args.dev_path)
    # restore the best model
    model.saver.restore(model.sess, save_path=model.save_path)

    # save as tf serving for online predict.
    serving_save_path = args.model_path
    with model.graph.as_default():
        if os.path.exists(serving_save_path):
            shutil.rmtree(serving_save_path)
        builder = tf.saved_model.builder.SavedModelBuilder(serving_save_path)
        inputs = {}
        for name in model.inputs:
            inputs[name] = tf.saved_model.utils.build_tensor_info(model.inputs[name])
        outputs = dict(zip(['pred{}:0'.format(task_idx) for task_idx in range(args.num_tasks)],
                           [tf.saved_model.utils.build_tensor_info(model.outputs['pred{}:0'.format(task_idx)]) for task_idx in range(args.num_tasks)]))

        signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs,
                                                                           tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(model.sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={'test_signature': signature},
                                             legacy_init_op=legacy_init_op)
        builder.save()
        print("done")

    print('loading tf-serving model from {}'.format(serving_save_path))

    with tf.Session() as sess:
        # load model
        meta_graph_def = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], serving_save_path)
        # get signature
        signature = meta_graph_def.signature_def
        key_my_signature = 'test_signature'
        # get tensor name
        train_ids = []
        for column in model.all_features:
            train_id = signature[key_my_signature].inputs['{}:0'.format(column)].name
            train_ids.append(sess.graph.get_tensor_by_name(train_id))
        for drop_idx in range(len(eval(args.keep_prob))):
            train_ids.append(sess.graph.get_tensor_by_name(signature[key_my_signature].inputs['dropout_keep{}:0'.format(drop_idx)].name))
        train_ids.append(sess.graph.get_tensor_by_name(signature[key_my_signature].inputs['train_phase:0'].name))
        preds = []
        for task_idx in range(args.num_tasks):
            preds.append(sess.graph.get_tensor_by_name(signature[key_my_signature].outputs['pred{}:0'.format(task_idx)].name))

        # Evaluate on validation data
        t = time.time()
        dev_gen = model.iterator(args.dev_path)
        true_pred = model.evaluate_generator_serving(sess, dev_gen, inputs=train_ids, outputs=preds)
        dev_result = model.metrics(true_pred)
        print_info('Serving DEV ', dev_result, time.time() - t)

        if len(args.test_path) > 0:
            # Evaluate on test data
            t = time.time()
            test_gen = model.iterator(args.test_path)
            true_pred = model.evaluate_generator_serving(sess, test_gen, inputs=train_ids, outputs=preds)
            test_result = model.metrics(true_pred)
            print_info('Serving Test ', test_result, time.time() - t)
