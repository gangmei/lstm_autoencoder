import io
from locale import normalize
import re
import string
from matplotlib.pyplot import axis
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import mpu
from sklearn.model_selection import train_test_split

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

"""### Configure the dataset for performance

To perform efficient batching for the potentially large number of training examples, use the `tf.data.Dataset` API. After this step, you would have a `tf.data.Dataset` object of `(target_word, context_word), (label)` elements to train your word2vec model!
"""
load_data = mpu.io.read('word2vec.pickle')
use_pretrained = True

targets, contexts = load_data['targets'], load_data['contexts']


vocab_size = np.max(contexts) + 1
contexts = contexts[:, 0]
BATCH_SIZE = 1024

test_ratio = 0.10

# train is now 90% of the entire data set
train_targets, test_targets, train_contexts, test_contexts = train_test_split(
    targets, contexts, test_size=test_ratio, random_state=42)


"""## Model and training

The word2vec model can be implemented as a classifier to distinguish between true context words from skip-grams and false context words obtained through negative sampling. You can perform a dot product multiplication between the embeddings of target and context words to obtain predictions for labels and compute the loss function against true labels in the dataset.

### Subclassed word2vec model

Use the [Keras Subclassing API](https://www.tensorflow.org/guide/keras/custom_layers_and_models) to define your word2vec model with the following layers:

* `target_embedding`: A `tf.keras.layers.Embedding` layer, which looks up the embedding of a word when it appears as a target word. The number of parameters in this layer are `(vocab_size * embedding_dim)`.
* `context_embedding`: Another `tf.keras.layers.Embedding` layer, which looks up the embedding of a word when it appears as a context word. The number of parameters in this layer are the same as those in `target_embedding`, i.e. `(vocab_size * embedding_dim)`.
* `dots`: A `tf.keras.layers.Dot` layer that computes the dot product of target and context embeddings from a training pair.
* `flatten`: A `tf.keras.layers.Flatten` layer to flatten the results of `dots` layer into logits.

With the subclassed model, you can define the `call()` function that accepts `(target, context)` pairs which can then be passed into their corresponding embedding layer. Reshape the `context_embedding` to perform a dot product with `target_embedding` and return the flattened result.

Key point: The `target_embedding` and `context_embedding` layers can be shared as well. You could also use a concatenation of both embeddings as the final word2vec embedding.
"""


class Word2Vec_SimCLR(models.Model):
    def __init__(self, vocab_size, embedding_dim, temperature=1.0, normalize=True):
        super(Word2Vec_SimCLR, self).__init__()
        self.target_embedding = layers.Embedding(vocab_size,
                                                 embedding_dim,
                                                 input_length=1,
                                                 name="w2v_embedding")

        self.temperature = temperature
        self.normalize = normalize

        self.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(
                         from_logits=True),
                     metrics=['accuracy'])

    def call(self, pair):
        target, context = pair
        batch_size = target.shape[0]

        latent_a = self.target_embedding(target)
        latent_b = self.target_embedding(context)

        if self.normalize:
            latent_a = tf.math.l2_normalize(latent_a, axis=-1)
            latent_b = tf.math.l2_normalize(latent_b, axis=-1)

        corr_aa = tf.matmul(latent_a, latent_a, transpose_b=True)
        corr_ab = tf.matmul(latent_a, latent_b, transpose_b=True)
        corr_bb = tf.matmul(latent_b, latent_b, transpose_b=True)
        corr_ba = tf.matmul(latent_b, latent_a, transpose_b=True)

        corr_neg = tf.eye(batch_size) * 1e6
        corr_aa = layers.subtract([corr_aa, corr_neg])
        corr_bb = layers.subtract([corr_bb, corr_neg])

        logits_a = layers.concatenate([corr_ab, corr_aa])
        logits_b = layers.concatenate([corr_bb, corr_ba])

        logits_a = logits_a/self.temperature
        logits_b = logits_b/self.temperature

        output = layers.concatenate([logits_a, logits_b], axis=0)

        return output

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, _ = data
        y = tf.eye(x[0].shape[0]*2)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, _ = data
        y = tf.eye(x[0].shape[0]*2)
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def build_graph(self):
        target = tf.keras.Input(shape=(1,))
        context = tf.keras.Input(shape=(1,))
        output = self.call((target, context))
        model = tf.keras.Model(
            inputs=[target, context], outputs=output, name="word2vec_model")
        return model

    def save_model(self, path_to_file: str = 'word2vec_model.h5'):
        self.save_weights(path_to_file)

    def load_model(self, path_to_file: str = 'word2vec_model.h5'):
        self.predict(
            (np.random.rand(5, 1), np.random.rand(5, 1)))
        self.load_weights(path_to_file)


"""### Define loss function and compile model

For simplicity, you can use `tf.keras.losses.CategoricalCrossEntropy` as an alternative to the negative sampling loss. If you would like to write your own custom loss function, you can also do so as follows:

``` python
def custom_loss(x_logit, y_true):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)
```

It's time to build your model! Instantiate your word2vec class with an embedding dimension of 128 (you could experiment with different values). Compile the model with the `tf.keras.optimizers.Adam` optimizer.
"""

embedding_dim = 128
word2vec = Word2Vec_SimCLR(vocab_size, embedding_dim,
                           temperature=0.1, normalize=True)
word2vec.run_eagerly = True


"""Also define a callback to log training statistics for Tensorboard:"""
"""Train the model on the `dataset` for some number of epochs:"""
if use_pretrained:
    word2vec.load_model()
else:
    train_labels = np.zeros_like(train_targets)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    word2vec.fit((train_targets, train_contexts), train_labels, batch_size=BATCH_SIZE, epochs=20, validation_split=0.05,
                 callbacks=[tensorboard_callback])
    word2vec.save_model()

test_labels = np.zeros_like(test_targets)

_, test_acc = word2vec.evaluate(
    (test_targets, test_contexts), test_labels, batch_size=BATCH_SIZE, verbose=0)
"""Tensorboard now shows the word2vec model's accuracy and loss:"""

print('test accuracy', test_acc)
