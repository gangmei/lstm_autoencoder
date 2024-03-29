import io
import re
import string
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
targets, contexts, labels = load_data['targets'], load_data['contexts'], load_data['labels']

vocab_size = np.max(contexts) + 1
num_ns = labels.shape[-1] - 1
BATCH_SIZE = 1024
BUFFER_SIZE = 10000

train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

# train is now 75% of the entire data set
# the _junk suffix means that we drop that variable completely
all_idx = np.arange(len(targets))
train_idx, test_idx = train_test_split(
    all_idx, test_size=1 - train_ratio)

val_idx, test_idx = train_test_split(
    test_idx, test_size=test_ratio/(test_ratio + validation_ratio))


train_dataset = tf.data.Dataset.from_tensor_slices(
    ((targets[train_idx], contexts[train_idx]), labels[train_idx]))
val_dataset = tf.data.Dataset.from_tensor_slices(
    ((targets[val_idx], contexts[val_idx]), labels[val_idx]))
test_dataset = tf.data.Dataset.from_tensor_slices(
    ((targets[test_idx], contexts[test_idx]), labels[test_idx]))

train_dataset = train_dataset.shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(train_dataset)

val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)


"""Apply `Dataset.cache` and `Dataset.prefetch` to improve performance:"""

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
print(train_dataset)

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


class Word2Vec(models.Model):
    def __init__(self, vocab_size, embedding_dim, temperature=1.0):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(vocab_size,
                                                 embedding_dim,
                                                 input_length=1,
                                                 name="w2v_embedding")
        self.context_embedding = layers.Embedding(vocab_size,
                                                  embedding_dim,
                                                  input_length=num_ns+1)
        self.temperature = temperature

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)

        word_emb = tf.math.l2_normalize(word_emb, -1)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        context_emb = tf.math.l2_normalize(context_emb, -1)

        # context_emb: (batch, context, embed)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)

        dots = dots / self.temperature
        # dots: (batch, context)
        return dots


"""### Define loss function and compile model

For simplicity, you can use `tf.keras.losses.CategoricalCrossEntropy` as an alternative to the negative sampling loss. If you would like to write your own custom loss function, you can also do so as follows:

``` python
def custom_loss(x_logit, y_true):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)
```

It's time to build your model! Instantiate your word2vec class with an embedding dimension of 128 (you could experiment with different values). Compile the model with the `tf.keras.optimizers.Adam` optimizer.
"""

embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(
                     from_logits=True),
                 metrics=['accuracy'])
# word2vec.run_eagerly = True

"""Also define a callback to log training statistics for Tensorboard:"""

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

"""Train the model on the `dataset` for some number of epochs:"""

word2vec.fit(train_dataset, epochs=20, validation_data=val_dataset,
             callbacks=[tensorboard_callback])

_, test_acc = word2vec.evaluate(
    (targets[test_idx], contexts[test_idx]), labels[test_idx], batch_size=BATCH_SIZE, verbose=0)
"""Tensorboard now shows the word2vec model's accuracy and loss:"""
print('test accuracy', test_acc)
