import tensorflow as tf
import os

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
    tpu="grpc://" + os.environ["COLAB_TPU_ADDR"]
)

tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

strategy = tf.distribute.TPUStrategy(resolver)


def create_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(256, 3, activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(256, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )


with strategy.scope():
    model = create_model()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )
