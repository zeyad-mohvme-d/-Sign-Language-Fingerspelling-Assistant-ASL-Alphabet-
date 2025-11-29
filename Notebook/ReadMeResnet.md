import tensorflow as tf

# Base ResNet50
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)
)
base_model.trainable = False

# Input layer
inputs = tf.keras.Input(shape=(224,224,3))

# Preprocessing / resizing (لو بياناتك أصلاً مش 224)
x = tf.keras.layers.Resizing(224,224)(inputs)

# Base model
x = base_model(x)

# Pooling + Dropout + Dense
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(29, activation="softmax")(x)

# Final model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
