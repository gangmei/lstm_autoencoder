import io
from locale import normalize
import re
import string
from matplotlib import path
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import mpu
from sklearn.model_selection import train_test_split

SEED = 42
"""### Configure the dataset for performance

To perform efficient batching for the potentially large number of training examples, use the `tf.data.Dataset` API. After this step, you would have a `tf.data.Dataset` object of `(target_word, context_word), (label)` elements to train your word2vec model!
"""
load_data = mpu.io.read('word2vec_onlinenew_neg100.pickle')
use_pretrained = False
use_base_weights = True
finetune_base_weights = True

load_model_path = 'word2vec_model_onlinebase.h5'
load_base_weights_path = 'word2vec_embedding_onlinebase.npy'


targets, contexts, labels = load_data['targets'], load_data['contexts'], load_data['labels']
# targets = np.expand_dims(targets, axis=-1)
contexts = np.squeeze(contexts)
# vocab_size = np.max(contexts) + 1
vocab_size = 4096
num_ns = labels.shape[-1] - 1
# seperate out base corpus and new corpus

BATCH_SIZE = 1024
BUFFER_SIZE = 10000
test_ratio = 0.10

# train is now 90% of the entire data set
train_targets, test_targets, train_contexts, test_contexts, train_labels, test_labels = train_test_split(
    targets, contexts, labels, test_size=test_ratio, random_state=42)


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


class Word2Vec_Online(models.Model):
    def __init__(self, vocab_size, base_vocab_size, embedding_dim, temperature=1.0, context_dim=5, normalize=True, share_emebdding=False):
        super(Word2Vec_Online, self).__init__()
        self.base_target_embedding = layers.Embedding(base_vocab_size,
                                                      embedding_dim,
                                                      input_length=1,
                                                      name="w2v_base_embedding")
        self.online_target_embedding = layers.Embedding(vocab_size - base_vocab_size,
                                                        embedding_dim,
                                                        input_length=1,
                                                        name="w2v_online_embedding")

        self.temperature = temperature
        self.context_dim = context_dim
        self.normalize = normalize
        self.base_vocab_size = base_vocab_size
        self.vocab_size = vocab_size

        # self.base_target_embedding.build(tf.TensorShape([None, 1]))
        # self.online_target_embedding.build(tf.TensorShape([None, context_dim]))

    def online_embed(self, input_tensor):
        """ method to apply online embedding with an input tensor. 
        For tokens within base vocab (id < base_vocab_size), apply base_target_embedding.
        For new tokens (id >= base_vocab_size, apply online_target_embedding) 

        Args:
            input_tensor ([type]): input tensor, target or context

        Returns:
            [type]: embedding output tensor for the input tensor
        """
        if len(input_tensor.shape) < 2:
            input_tensor = tf.expand_dims(input_tensor, axis=-1)
        input_mask = tf.cast(
            input_tensor < self.base_vocab_size, dtype=tf.int64)

        base_output = self.base_target_embedding(input_mask * input_tensor)
        input_mask2 = tf.repeat(tf.expand_dims(
            input_mask, axis=-1), base_output.shape[-1], axis=2)
        base_output_masked = layers.multiply(
            [base_output, tf.cast(input_mask2, dtype=tf.float32)])

        online_output = self.online_target_embedding(
            (1-input_mask) * (input_tensor-self.base_vocab_size))
        online_output_masked = layers.multiply(
            [online_output, tf.cast(1-input_mask2, dtype=tf.float32)])

        return base_output_masked + online_output_masked

    def call(self, pair):
        target, context = pair

        word_emb = self.online_embed(target)
        context_emb = self.online_embed(context)

        dots = layers.Dot(
            axes=-1, normalize=self.normalize)([context_emb, word_emb])
        dots = layers.Flatten()(dots)
        dots = dots / self.temperature

        return dots

    def build_graph(self):
        target = tf.keras.Input(shape=(1,), dtype=tf.int64)
        context = tf.keras.Input(shape=(self.context_dim,), dtype=tf.int64)
        output = self.call((target, context))
        model = tf.keras.Model(
            inputs=[target, context], outputs=output, name="word2vec_model")
        return model

    def save_model(self, path_to_file: str = 'word2vec_model.h5'):
        self.save_weights(path_to_file)

    def load_model(self, path_to_file: str = 'word2vec_model.h5'):
        # self.predict(
        #     (np.random.randint(self.vocab_size, size=(5, 1)), np.random.randint(self.vocab_size, size=(5, self.context_dim))))
        self.load_weights(path_to_file)

    def load_base_weights(self, path_to_file: str = 'word2vec_embeddings.npy', fine_tune_enabled=False):
        self.predict(
            (np.random.randint(self.vocab_size, size=(5, 1)), np.random.randint(self.vocab_size, size=(5, self.context_dim))))
        base_embedding_weights = np.load(path_to_file)
        self.base_target_embedding.set_weights([base_embedding_weights])
        self.base_target_embedding.trainable = fine_tune_enabled


"""### Define loss function and compile model

For simplicity, you can use `tf.keras.losses.CategoricalCrossEntropy` as an alternative to the negative sampling loss. If you would like to write your own custom loss function, you can also do so as follows:

``` python
def custom_loss(x_logit, y_true):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)
```

It's time to build your model! Instantiate your word2vec class with an embedding dimension of 128 (you could experiment with different values). Compile the model with the `tf.keras.optimizers.Adam` optimizer.
"""

embedding_dim = 128
word2vec = Word2Vec_Online(vocab_size, 3800, embedding_dim,
                           context_dim=num_ns+1, temperature=0.1, normalize=True, share_emebdding=True)
# word2vec.run_eagerly = True


"""Also define a callback to log training statistics for Tensorboard:"""
"""Train the model on the `dataset` for some number of epochs:"""
if use_pretrained:
    word2vec.load_model(load_model_path)
else:
    if use_base_weights:
        word2vec.load_base_weights(
            load_base_weights_path, finetune_base_weights)

# tf.keras.utils.plot_model(word2vec.build_graph(), show_shapes=True)

word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(
                     from_logits=True),
                 metrics=['accuracy'])

_, test_acc = word2vec.evaluate(
    (test_targets, test_contexts), test_labels, batch_size=BATCH_SIZE, verbose=0)

print('test accuracy before training', test_acc)

tf.keras.utils.plot_model(word2vec.build_graph(), show_shapes=True)


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
word2vec.fit((train_targets, train_contexts), train_labels, batch_size=BATCH_SIZE, epochs=20, validation_split=0.05,
             callbacks=[tensorboard_callback])
word2vec.save_model()

_, test_acc = word2vec.evaluate(
    (test_targets, test_contexts), test_labels, batch_size=BATCH_SIZE, verbose=0)
"""Tensorboard now shows the word2vec model's accuracy and loss:"""

print('test accuracy', test_acc)
