import numpy as np
import tensorflow as tf
import random, re

from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.utils import plot_model

from my_transformer import PositionalEmbedding, TransformerEncoder, TransformerDecoder
from my_sent_tokenizer import text_vectorization, text_vectorization_tfds


max_len = 20
vocab_size = 20000
embed_dim = 256
num_heads = 8
dense_dim = 1024

epochs = 1
batch_size = 128
buffer_size = 20000

(
    train_ds,
    valid_ds,
    test_pairs,
    vocab_size,
    tokenizer,
    start_token,
    end_token,
) = text_vectorization_tfds(max_len=max_len, vocab_size=vocab_size, batch_size=batch_size)
print(vocab_size)


encoder_inputs = Input(shape=(None,), dtype="int64", name="source")
x = PositionalEmbedding(max_len, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

decoder_inputs = Input(shape=(None,), dtype="int64", name="target")
x = PositionalEmbedding(max_len, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
x = Dropout(0.5)(x)
decoder_outputs = Dense(vocab_size, activation="softmax")(x)
transformer = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
plot_model(transformer, "images/my_transformer_bot_tfds.png", show_shapes=True)
transformer.summary()


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
transformer.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

callbacks = [keras.callbacks.ModelCheckpoint("../data/my_transformer_bot.tfds")]
transformer.fit(train_ds, epochs=epochs, validation_data=valid_ds)

# transformer = keras.models.load_model(
#     "../data/my_transformer_bot.tfds",
#     custom_objects={
#         "PositionalEmbedding": PositionalEmbedding,
#         "TransformerEncoder": TransformerEncoder,
#         "TransformerDecoder": TransformerDecoder,
#     },
# )


def decode_sequence(sentence):
    # sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    # sentence = sentence.strip()
    tokenized_sentence = tf.expand_dims(
        start_token + tokenizer.encode(sentence) + end_token, axis=0
    )
    decoded_sentence = tf.expand_dims(start_token, axis=0)
    for i in range(max_len):
        tokenized_target_sentence = tokenizer.decode([decoded_sentence])
        tokenized_target_sentence = tokenized_target_sentence[:, :-1]
        predictions = transformer([tokenized_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = tokenizer.decode(sampled_token_index)
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence


test_texts = [(pair[0], pair[1]) for pair in test_pairs]
for _ in range(1):
    input_sentence = random.choice(test_texts)
    print("-" * 90)
    print(input_sentence[0])
    print(input_sentence[1])
    print(decode_sequence(input_sentence[0]))


def evaluate(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    sentence = tf.expand_dims(start_token + tokenizer.encode(sentence) + end_token, axis=0)
    output = tf.expand_dims(start_token, 0)

    for i in range(max_len):
        predictions = transformer(inputs=[sentence, output], training=False)
        predictions = predictions[:, i, :]
        # predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if tf.equal(predicted_id, end_token[0]):
            break
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)
    predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])
    print("Input: {}".format(sentence))
    print("Output: {}".format(predicted_sentence))
    return predicted_sentence


# predict("?????? ???????")
# predict("????????? ??????")
# predict("?????? ?????????")
# predict("????????????????????? ???????")
# predict("??? ??? ???????????? ??? ??????")
# predict("??? ?????? ????????? ????????? ??? ?????? ??????")
