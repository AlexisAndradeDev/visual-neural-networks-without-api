import tensorflow as tf
import numpy as np

from app import db, STATIC_DIR
from app.playground.models import Layer, NeuralNetwork

DATASETS_NAMES = ["horse_or_human", "cat_or_dog"]
LAYER_TYPES = ["Dense", "Conv2D", "MaxPooling2D", "Flatten"]
OPTIMIZERS = {
    "Adam": tf.keras.optimizers.Adam,
    "SGD": tf.keras.optimizers.SGD,
    "RMSprop": tf.keras.optimizers.RMSprop,
}
LOSS_FUNCTIONS = [
    "binary_crossentropy", "sparse_categorical_crossentropy",
    "mean_squared_error",
]
ACTIVATION_FUNCTIONS = ["relu", "softmax", "sigmoid"]


def create_data_generators(dataset_name):
    assert dataset_name in DATASETS_NAMES, f"Dataset '{dataset_name}' does not exist."

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

    if dataset_name == "horse_or_human":
        train_generator = train_datagen.flow_from_directory(
            (STATIC_DIR + "/horse_or_human/train/").replace("\\", "/"),
            target_size=(300, 300), batch_size=128, class_mode="binary",
        )
        validation_generator = validation_datagen.flow_from_directory(
            (STATIC_DIR + "/horse_or_human/validation/").replace("\\", "/"),
            target_size=(300, 300), batch_size=32, class_mode="binary",
        )
    elif dataset_name == "cat_or_dog":
        train_generator = train_datagen.flow_from_directory(
            (STATIC_DIR + "/cat_or_dog/train/").replace("\\", "/"),
            target_size=(150, 150), batch_size=50, class_mode="binary",
        )
        validation_generator = validation_datagen.flow_from_directory(
            (STATIC_DIR + "/cat_or_dog/validation/").replace("\\", "/"),
            target_size=(150, 150), batch_size=50, class_mode="binary",
        )

    return train_generator, validation_generator

def create_layer(name, type_, neural_network_id, index, units=None, filters=None,
        activation_function=None, kernel_size=None, pool_size=None):
    """
    Creates an app.playground.models.Layer object and adds it to the
    database.

    Args:
        name (str): Name of the layer.
        type_ (str): Type of layer.
            Must be: Conv2D, Dense, MaxPooling2D or Flatten.
        neural_network_id (int): ID of the Neural Network that contains this layer.
        index (int): Index of the layer within the neural network layers.
        units (int, optional): Number of neurons.
            Layer types that use this parameter: Dense.
            Defaults to None.
        filters (int, optional): Number of filters.
            Layer types that use this parameter: Conv2D.
            Defaults to None.
        activation_function (str, optional): Activation function.
            Layer types that use this parameter: Dense, Conv2D.
            Must be: relu, softmax, sigmoid.
            activation_function in ACTIVATION_FUNCTIONS must be True.
            Defaults to None.
        kernel_size (int, optional): Size of the kernel.
            Layer types that use this parameter: Conv2D.
            Defaults to None.
        pool_size (int, optional): Size of the pooling window.
            Layer types that use this parameter: MaxPooling2D.
            Defaults to None.
    """
    assert type_ in LAYER_TYPES, f"Layer type '{type_}' does not exist."
    if activation_function:
        assert activation_function in ACTIVATION_FUNCTIONS, f"Activation function {activation_function} does not exist."

    layer = Layer(
        name=name, type_=type_, neural_network_id=neural_network_id, index=index, units=units,
        filters=filters, activation_function=activation_function,
        kernel_size=kernel_size, pool_size=pool_size,
    )
    db.session.add(layer)
    db.session.commit()

def create_default_layer(neural_network):
    last_layer_index = neural_network.sorted_layers[-1].index

    layer_index = last_layer_index + 1
    default_name = f"Layer {layer_index}"

    create_layer(
        name=default_name, type_="Dense",
        neural_network_id=neural_network.id, index=layer_index,
    )

def database_layer_to_keras_layer(layer):
    """
    Converts a app.playground.models.Layer object into a Keras layer object
    and returns it.

    Args:
        layer (app.playground.models.Layer): Layer object.

    Returns:
        keras_layer: Keras layer object.
    """
    if layer.type_ == "Dense":
        keras_layer = tf.keras.layers.Dense(
            layer.units, activation=layer.activation_function,
        )
    elif layer.type_ == "Conv2D":
        keras_layer = tf.keras.layers.Conv2D(
            layer.filters, layer.kernel_size,
            activation=layer.activation_function,
        )
    elif layer.type_ == "MaxPooling2D":
        keras_layer = tf.keras.layers.MaxPooling2D(layer.pool_size)
    elif layer.type_ == "Flatten":
        keras_layer = tf.keras.layers.Flatten()
    return keras_layer

def create_neural_network(name, loss, optimizer, epochs, dataset_name, user_id,
        lr=None):
    """
    Creates an app.playground.models.NeuralNetwork object and adds it to the
    database. Returns the NeuralNetwork object.

    Args:
        name (str): Name of the Neural Network.
        loss (str): Name of a loss function.
            Must be: binary_crossentropy, sparse_categorical_crossentropy,
            mean_squared_error.
            loss in LOSS_FUNCTIONS must be True.
            If you want to add a new loss function, modify the
            LOSS_FUNCTIONS list.
        optimizer (str): Name of an optimizer.
            Must be: Adam, SGD, RMSprop.
            If you want to add a new optimizer, modify the OPTIMIZERS dict.
        epochs (int): Number of epochs (iterations).
        dataset_name (str): Name of a dataset.
            dataset_name in DATASETS_NAMES must be True.
        user_id (int): ID of the user that created this Neural Network.
        lr (float): Learning rate.

    Returns:
        neural_network (app.playground.models.NeuralNetwork): Neural Network object.
    """
    assert loss in LOSS_FUNCTIONS, f"Loss function '{loss}' does not exist."
    assert optimizer in OPTIMIZERS, f"Optimizer '{optimizer}' does not exist."

    neural_network = NeuralNetwork(
        name=name, loss=loss, optimizer=optimizer, learning_rate=lr,
        epochs=epochs, dataset_name=dataset_name, creator_id=user_id,
    )
    db.session.add(neural_network)
    db.session.commit()
    return neural_network

def database_nn_to_keras_model(neural_network):
    keras_layers = []
    for layer in neural_network.sorted_layers:
        keras_layer = database_layer_to_keras_layer(layer)
        keras_layers.append(keras_layer)
    model = tf.keras.models.Sequential(keras_layers)

    # create instance of the optimizer
    if neural_network.optimizer in ["RMSprop"]:
        optimizer = OPTIMIZERS[neural_network.optimizer](neural_network.learning_rate)
    else:
        optimizer = OPTIMIZERS[neural_network.optimizer]()

    model.compile(loss=neural_network.loss,
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model

def train_model(model, epochs, X_train=None, y_train=None,
        X_test=None, y_test=None, train_generator=None,
        validation_generator=None):
    if train_generator and validation_generator:
        history = model.fit(
            train_generator, steps_per_epoch=8, epochs=epochs,
            validation_data=validation_generator, validation_steps=8,
        )
    elif X_train and y_train and X_test and y_test:
        history = model.fit(X_train, y_train, epochs=epochs)
        model.evaluate(X_test, y_test)
    else:
        raise AssertionError("Model training error. If you are use generators, specify both training and validation generators. If you use array datasets, specify X_train, y_train, X_test and y_test.")

    return history

def train_neural_network(neural_network):
    """
    Creates a Keras model with the input neural network, trains it,
    and returns the trained model.

    Args:
        neural_network (app.playground.models.NeuralNetwork): Neural Network object.

    Returns:
        model (tf.keras.models.Sequential): Trained Keras model.
        history: History of the metrics (loss, accuracy, etc.).
    """
    model = database_nn_to_keras_model(neural_network)
    train_generator, validation_generator = create_data_generators(
        neural_network.dataset_name,
    )
    history = train_model(
        model, neural_network.epochs, train_generator=train_generator,
        validation_generator=validation_generator,
    )
    return model, history

def get_intermediate_representations(model, neural_network):
    """
    Returns the intermediate representations of a random image from the
    visualization dataset. The intermediate representations are the
    images of the filters applied to the random image by the neural network.

    Args:
        model (tf.keras.models.Sequential): Trained Keras model.
        neural_network (app.playground.models.NeuralNetwork): Neural Network object.

    Returns:
        features_images (dict of list of numpy.ndarray): Images of the
            intermediate representations.
    """
    # create visualization model
    layers_outputs = [layer.output for layer in model.layers]
    visualization_model = tf.keras.models.Model(inputs=model.input, outputs=layers_outputs)

    # load an image
    if neural_network.dataset_name == "horse_or_human":
        img = tf.keras.preprocessing.image.load_img(
            STATIC_DIR + "/horse_or_human/visualization/horses/horse1.jpg",
            target_size=(300,300),
        )
    elif neural_network.dataset_name == "cat_or_dog":
        img = tf.keras.preprocessing.image.load_img(
            STATIC_DIR + "/cat_or_dog/visualization/cat/cat1.jpg",
            target_size=(150, 150),
        )
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255 # normalization

    # obtain intermediate representations
    features_maps = visualization_model.predict(x)

    # create representations images
    features_images = {}
    for layer, features_map in zip(neural_network.sorted_layers, features_maps):
        # display conv / maxpool layers
        if layer.type_ in ["Conv2D", "MaxPooling2D"]:
            features_number = features_map.shape[3]
            img_size = features_map.shape[1]

            features_images[layer.id] = []
            for feature_index in range(features_number):
                # process the feature image
                feature_img = features_map[0, :, :, feature_index]
                feature_img -= feature_img.mean()
                feature_img /= feature_img.std()
                feature_img *= 64
                feature_img += 128

                feature_img = np.clip(feature_img, 0, 255).astype("uint8")
                features_images[layer.id].append(feature_img)

    return features_images
