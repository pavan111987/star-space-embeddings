import tensorflow as tf
import numpy as np
import random
from scipy.spatial.distance import cosine as dist
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


class StarSpaceShip:

    def __init__(self):
        self.input_encoder_model = None
        self.target_encoder_model = None
        self.model = None

        self.target_encodings = None
        self.distance_dict = None

        self.test_positive_input_batches = None
        self.test_positive_batch_targets = None
        self.test_negative_batch_targets = None

    @staticmethod
    def _fetch_prepare_rawdata():

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])/255.
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/255.

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        return x_train, y_train, x_test, y_test

    @staticmethod
    def custom_loss(y_true, y_pred):

        pos_input_emb = y_pred[:, 0, :]
        pos_target_emb = y_pred[:, 1, :]
        neg_target_embs = y_pred[:, 2:, :]

        scores_pos = tf.expand_dims(tf.reduce_sum(tf.multiply(pos_input_emb, pos_target_emb), -1), axis = -1)
        scores_neg = tf.expand_dims(tf.reduce_sum(tf.multiply(pos_input_emb, tf.math.reduce_mean(neg_target_embs, axis = 1)), -1), axis = -1)

        loss_matrix = tf.maximum(0., 1. - scores_pos + scores_neg)
        loss = tf.reduce_sum(loss_matrix)

        return loss

    def prepare_features_targets(self):
        x_train, y_train, x_test, y_test = self._fetch_prepare_rawdata()

        train_positive_input_batches, train_positive_batch_targets, train_negative_batch_targets = self.generate_batches(x_train, y_train)
        test_positive_input_batches, test_positive_batch_targets, test_negative_batch_targets = self.generate_batches(x_test, y_test)

        train_dummy_outputs = np.zeros((len(train_positive_input_batches), 12, 256))
        test_dummy_outputs = np.zeros((len(test_positive_input_batches), 12, 256))

        train_positive_input_batches = train_positive_input_batches.reshape(len(train_positive_input_batches), 1, 784)
        train_positive_batch_targets = train_positive_batch_targets.reshape(len(train_positive_batch_targets), 1)
        train_negative_batch_targets = train_negative_batch_targets.reshape(len(train_negative_batch_targets), 10)

        test_positive_input_batches = test_positive_input_batches.reshape(len(test_positive_input_batches), 1, 784)
        test_positive_batch_targets = test_positive_batch_targets.reshape(len(test_positive_batch_targets), 1)
        test_negative_batch_targets = test_negative_batch_targets.reshape(len(test_negative_batch_targets), 10)

        return train_positive_input_batches, train_positive_batch_targets, train_negative_batch_targets, train_dummy_outputs,\
             test_positive_input_batches, test_positive_batch_targets, test_negative_batch_targets, test_dummy_outputs

    def generate_batches(self, x, y):
        positive_input_batches = []
        negative_input_batches = []
        positive_batch_targets = []
        negative_batch_targets = []

        for idx, x_feats in enumerate(x):

            positive_input_batch = [x[idx]]
            positive_batch_target = [y[idx]]
            negative_batch_target = []

            for neg_idx in random.sample(list(np.where(y[:, 0] != y[idx][0])[0]), 10):
                negative_batch_target.append(y[neg_idx])

            positive_input_batches.append(np.array(positive_input_batch))
            positive_batch_targets.append(np.array(positive_batch_target))
            negative_batch_targets.append(np.array(negative_batch_target))

        return np.array(positive_input_batches), np.array(positive_batch_targets), np.array(negative_batch_targets)

    def build_model_architecture(self):
        positive_input = tf.keras.layers.Input(shape = (1, 784))
        positive_target_input = tf.keras.layers.Input(shape = (1, ))
        negative_target_inputs = tf.keras.layers.Input(shape = (10, ))

        input_dense_layer = tf.keras.layers.Dense(256)
        target_embedding_layer = tf.keras.layers.Embedding(input_dim = 10, output_dim = 256)
        target_dense_layer = tf.keras.layers.Dense(256)

        pos_input_embedding = tf.nn.l2_normalize(input_dense_layer(positive_input), -1)

        pos_target_embedding = tf.nn.l2_normalize(target_dense_layer(target_embedding_layer(positive_target_input)), -1)
        neg_target_embedding = tf.nn.l2_normalize(target_dense_layer(target_embedding_layer(negative_target_inputs)), -1)

        packed_output_embeddings = tf.keras.layers.concatenate([pos_input_embedding, pos_target_embedding, neg_target_embedding], axis = 1)

        self.model = tf.keras.models.Model(inputs = [positive_input, positive_target_input, negative_target_inputs], outputs = packed_output_embeddings)

        self.model.compile(loss = self.custom_loss, optimizer = 'adam')

        self.input_encoder_model = tf.keras.models.Model(inputs = positive_input, outputs = pos_input_embedding)
        self.target_encoder_model = tf.keras.models.Model(inputs = positive_target_input, outputs = pos_target_embedding)

    def train_star_space(self):
        train_positive_input_batches, train_positive_batch_targets, train_negative_batch_targets, train_dummy_outputs,\
             test_positive_input_batches, test_positive_batch_targets, test_negative_batch_targets, test_dummy_outputs = self.prepare_features_targets()

        self.test_positive_input_batches = test_positive_input_batches
        self.test_positive_batch_targets = test_positive_batch_targets
        self.test_negative_batch_targets = test_negative_batch_targets

        self.build_model_architecture()
        print (self.model.summary())

        self.model.fit([train_positive_input_batches, train_positive_batch_targets, train_negative_batch_targets], train_dummy_outputs, epochs = 10,\
         validation_data = ([test_positive_input_batches, test_positive_batch_targets, test_negative_batch_targets], test_dummy_outputs))

        self.target_encodings = {target_id: self.target_encoder_model.predict(np.array([target_id]))[0, 0, :] for target_id in range(10)}

        d = {}
        for i in range(10):
            for j in range(10):
                if i != j and f'{j}_{i}' not in d:
                    d[f'{i}_{j}'] = 1 - dist(self.target_encodings[i], self.target_encodings[j])

        print ({k: v for k, v in sorted(d.items(), key=lambda item: item[1])})

    def predict_class(self, input_image = None):

        actual_target = None
        if input_image is None and self.test_positive_input_batches is not None:
            random_idx = np.random.randint(0, len(self.test_positive_input_batches))
            input_image = self.test_positive_input_batches[random_idx, :].reshape(1, 1, 784)
            actual_target = self.test_positive_batch_targets[random_idx]

        if self.input_encoder_model:
            input_encodings = self.input_encoder_model.predict(input_image)[0, 0, :]

        distance_dict = {i: 1.0 - dist(self.target_encodings[i], input_encodings) for i in range(10)}
        
        return distance_dict, actual_target

    def save_projector_tensorflow_files(self):
        '''
        This method prepares the held-out set to enable PCA visualization using https://projector.tensorflow.org/
        To do this we need two TSV files one containing the floating point tab serperated embeddings file
        The other file has labels for each of the rows
        '''
        testset_embeddings = self.input_encoder_model.predict(self.test_positive_input_batches)
        testset_embeddings = testset_embeddings[:, 0, :].astype('U25')
        
        with open("visualization/projector_tensorflow_data/test_embedding_vectors.tsv", "w") as f:
            f.write("\n".join(["\t".join(testset_embedding) for testset_embedding in testset_embeddings]))

        f.close()

        with open("visualization/projector_tensorflow_data/test_embedding_labels.tsv", "w") as f:
            f.write("\n".join([str(label[0]) for label in self.test_positive_batch_targets]))

        f.close()

    def plot_confusion_matrix(self):
        labels = list(range(10))
        y_pred = []
        y_true = []
        for idx, test_positive_input in enumerate(self.test_positive_input_batches):
            scores = [score for _, score in self.predict_class(test_positive_input.reshape(1, 1, 784))[0].items()]
            y_pred.append(np.argmax(scores))
            y_true.append(self.test_positive_batch_targets[idx][0])

        cm = confusion_matrix(y_true, y_pred, labels)
        
        labels = [str(label) for label in labels]

        plt.imshow(cm, interpolation='nearest')
        plt.xticks(np.arange(0, 10), labels)
        plt.yticks(np.arange(0, 10), labels)

        plt.savefig("visualization/confusion_matrix.png")
