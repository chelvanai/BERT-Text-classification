import re
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import tensorflow_hub as hub
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.preprocessing import sequence


# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {"sentence": []}
    for file_path in os.listdir(directory):
        with open(directory + "/" + file_path) as f:
            text = f.read()
            words = set(tf.keras.preprocessing.text.text_to_word_sequence(text))
            vocab_size = len(words)
            result = tf.keras.preprocessing.text.one_hot(text, round(vocab_size * 1.3))
            data['sentence'].append(result)

    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset():
    pos_df = load_directory_data('Data/train/pos')
    neg_df = load_directory_data('Data/train/neg')
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df])


x_train = load_dataset()['sentence'].values
y_train = load_dataset()['polarity'].values

x_train = sequence.pad_sequences(x_train, maxlen=2000)
y_train = utils.to_categorical(y_train, 2)


# Create custom bert layer
class BertLayer(tf.keras.layers.Layer):
    def __init__(self, n_fine_tune_layers=10, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            bert_path,
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )
        trainable_vars = self.bert.variables

        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]

        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        # Add non-trainable weights
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            "pooled_output"
        ]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


# Build model
in_id = tf.keras.layers.Input(shape=(2000,), name="input_ids")
in_mask = tf.keras.layers.Input(shape=(2000,), name="input_masks")
in_segment = tf.keras.layers.Input(shape=(2000,), name="segment_ids")
bert_inputs = [in_id, in_mask, in_segment]

# Instantiate the custom Bert Layer defined above
bert_output = BertLayer(n_fine_tune_layers=10)(bert_inputs)

# Build the rest of the classifier
dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)
pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# model.fit(
#     [train_input_ids, train_input_masks, train_segment_ids],
#     train_labels,
#     validation_data=([test_input_ids, test_input_masks, test_segment_ids], test_labels),
#     epochs=1,
#     batch_size=32
# )
