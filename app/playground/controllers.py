import io
import base64

from flask import Blueprint, render_template, request, redirect, flash
from PIL import Image
from flask.helpers import url_for

from app.playground.forms import SelectNeuralNetworkForm, NeuralNetworkParametersForm, TrainForm, LayersParametersForm, CreateLayerForm
from app.playground import deep_learning_funcs
from app.playground.models import NeuralNetwork, db
from app.playground.validators import DataRequired, DataRequiredIfDataInOtherFieldInValues, validate_field, Number, NoDashes, InList

playground_bp = Blueprint("playground", __name__, url_prefix="/playground")


def to_number_or_None(var, type_):
    """
    Returns var as int or float if var is not None, 'None' or empty 
    string, else returns None.

    Args:
        var (int, None): str or None var.
        type (str): 'float' or 'int'.

    Raises:
        AssertionError: If type_ is not int or float.

    Returns:
        int(var) if var is not None and type_ is int,
        float(var) if var is not None and type_ is float,
        else returns None.
    """
    if var is not None and var not in ["None", ""]:
        if type_ == "int":
            new_var = int(var)
        elif type_ == "float":
            new_var = float(var)
        else:
            raise AssertionError(f"Can not convert to type '{type_}'.")
    else:
        new_var = None
    return new_var

@playground_bp.route("/", methods=["GET", "POST"])
@playground_bp.route("/<neural_network_name>", methods=["GET", "POST"])
def main_page(neural_network_name=None):
    select_nn_form = SelectNeuralNetworkForm()
    nn_parameters_form = NeuralNetworkParametersForm()
    layers_parameters_form = LayersParametersForm()
    create_layer_form = CreateLayerForm()
    train_form = TrainForm()

    neural_networks = NeuralNetwork.query.all()

    if not neural_network_name:
        neural_network = neural_networks[-1] # default
        return redirect(url_for("playground.main_page", neural_network_name=neural_network.name))
    else:
        neural_network = NeuralNetwork.query.filter_by(name=neural_network_name).first()

    metrics_history = {}
    layers_images_base64 = {}
    show_results = False

    # if request.method == "GET":
    #     neural_network = deep_learning_funcs.create_neural_network(
    #         name="Cat or dog", loss="binary_crossentropy", optimizer="Adam",
    #         epochs=2, dataset_name="cat_or_dog", user_id=1, lr=None,
    #     )

    #     deep_learning_funcs.create_layer(
    #         name="Convolution 1", type_="Conv2D",
    #         neural_network_id=neural_network.id, index=0, filters=16, kernel_size=3,
    #         activation_function="relu",
    #     )
    #     deep_learning_funcs.create_layer(
    #         name="Max Pooling 1", type_="MaxPooling2D",
    #         neural_network_id=neural_network.id, index=1, pool_size=4,
    #     )
    #     deep_learning_funcs.create_layer(
    #         name="Convolution 2", type_="Conv2D",
    #         neural_network_id=neural_network.id, index=2, filters=16, kernel_size=3,
    #         activation_function="relu",
    #     )
    #     deep_learning_funcs.create_layer(
    #         name="Max Pooling 2", type_="MaxPooling2D",
    #         neural_network_id=neural_network.id, index=3, pool_size=4,
    #     )
    #     deep_learning_funcs.create_layer(
    #         name="Flatten", type_="Flatten", neural_network_id=neural_network.id,
    #         index=4,
    #     )
    #     deep_learning_funcs.create_layer(
    #         name="Dense1", type_="Dense", neural_network_id=neural_network.id,
    #         index=5, units=28, activation_function="relu",
    #     )
    #     deep_learning_funcs.create_layer(
    #         name="Dense2", type_="Dense", neural_network_id=neural_network.id,
    #         index=6, units=1, activation_function="sigmoid",
    #     )

    if request.method == "POST":
        if select_nn_form.submitted():
            neural_network = NeuralNetwork.query.filter_by(
                name=request.form.get("selected_neural_network")).first()
            return redirect(url_for("playground.main_page", neural_network_name=neural_network.name))

        if nn_parameters_form.submitted():
            validate_failed = False

            neural_network.name, validate_failed = validate_field(
                field_value=request.form.get("neural_network_name"), 
                previous_value=neural_network.name, 
                field_name="Neural Network Name", 
                validate_failed=validate_failed,
                validators=[DataRequired(), NoDashes()],
            )

            neural_network.loss, validate_failed = validate_field(
                field_value=request.form.get("loss_function"), 
                previous_value=neural_network.loss,
                field_name="Loss Function",
                validate_failed=validate_failed,
                validators=[DataRequired(), 
                    InList(deep_learning_funcs.LOSS_FUNCTIONS),
                ],
            )

            neural_network.optimizer, validate_failed = validate_field(
                field_value=request.form.get("optimizer"),
                previous_value=neural_network.optimizer,
                field_name="Optimizer",
                validate_failed=validate_failed,
                validators=[DataRequired(), 
                    InList(deep_learning_funcs.OPTIMIZERS),
                ],
            )

            neural_network.learning_rate, validate_failed = validate_field(
                field_value=request.form.get("learning_rate"), 
                previous_value=neural_network.learning_rate,
                field_name="Learning rate",
                validate_failed=validate_failed,
                validators=[Number(),
                    DataRequiredIfDataInOtherFieldInValues(
                        other_field_value=request.form.get("optimizer"),
                        possible_values=["RMSprop"],
                    ),
                ],
            )
            neural_network.learning_rate = to_number_or_None(
                neural_network.learning_rate, type_="float",
            )

            neural_network.epochs, validate_failed = validate_field(
                field_value=request.form.get("epochs"), 
                previous_value=neural_network.epochs,
                field_name="Epochs / Iterations",
                validate_failed=validate_failed,
                validators=[DataRequired(), Number(type_="int")],
            )
            neural_network.epochs = to_number_or_None(neural_network.epochs, type_="int")

            neural_network.dataset_name, validate_failed = validate_field(
                field_value=request.form.get("dataset_name"), 
                previous_value=neural_network.dataset_name,
                field_name="Dataset",
                validate_failed=validate_failed,
                validators=[DataRequired(), 
                    InList(deep_learning_funcs.DATASETS_NAMES)
                ],
            )

            db.session.commit()

            if validate_failed:
                flash("Some changes were not saved.", category="warning")
                return redirect(
                    url_for("playground.main_page", neural_network_name=neural_network.name))

        if layers_parameters_form.submitted():
            layers_order = request.form.get(f"layers_order").split(",")
            layers_order = [int(layer_id) for layer_id in layers_order]

            validate_failed = False
            for layer in neural_network.sorted_layers:
                delete_layer = bool(request.form.get(f"delete_{layer.id}"))
                if delete_layer:
                    db.session.delete(layer)
                else:
                    # name field
                    layer.name, validate_failed = validate_field(
                        field_value = request.form.get(f"name_{layer.id}"),
                        previous_value=layer.name,
                        field_name=f"Name. Layer: {layer.name}.",
                        validate_failed=validate_failed,
                        validators=[DataRequired(), NoDashes()],
                    )

                    # type field
                    layer.type_, validate_failed = validate_field(
                        previous_value=layer.type_,
                        field_value=request.form.get(f"type_{layer.id}"),
                        field_name=f"Type. Layer: {layer.name}",
                        validate_failed=validate_failed,
                        validators=[
                            NoDashes(), 
                            InList(deep_learning_funcs.LAYER_TYPES),
                        ],
                    )

                    layer.index = layers_order.index(layer.id)

                    # units field
                    layer.units, validate_failed = validate_field(
                        field_value=request.form.get(f"units_{layer.id}"),
                        previous_value=layer.units,
                        field_name=f"Units. Layer: {layer.name}",
                        validate_failed=validate_failed,
                        validators=[
                            NoDashes(), Number(type_="int"), 
                            DataRequiredIfDataInOtherFieldInValues(
                                other_field_value=layer.type_, 
                                possible_values=["Dense"],
                            ),
                        ],
                    )
                    layer.units = to_number_or_None(layer.units, type_="int")

                    # filters field
                    layer.filters, validate_failed = validate_field(
                        field_value=request.form.get(f"filters_{layer.id}"),
                        previous_value=layer.filters,
                        field_name=f"Number of filters. Layer: {layer.name}",
                        validate_failed=validate_failed,
                        validators=[
                            NoDashes(), Number(type_="int"),
                            DataRequiredIfDataInOtherFieldInValues(
                                other_field_value=layer.type_,
                                possible_values=["Conv2D"],
                            ),
                        ],
                    )
                    layer.filters = to_number_or_None(layer.filters, type_="int")

                    # activation function field
                    layer.activation_function, validate_failed = validate_field(
                        field_value=request.form.get(f"activation_function_{layer.id}"),
                        previous_value=layer.activation_function,
                        field_name=f"Activation function. Layer: {layer.name}",
                        validate_failed=validate_failed,
                        validators=[
                            NoDashes(),
                            InList(deep_learning_funcs.ACTIVATION_FUNCTIONS),
                            DataRequiredIfDataInOtherFieldInValues(
                                other_field_value=layer.type_,
                                possible_values=["Dense", "Conv2D"],
                            ),
                        ],
                    )

                    # kernel size field
                    layer.kernel_size, validate_failed = validate_field(
                        field_value=request.form.get(f"kernel_size_{layer.id}"),
                        previous_value=layer.kernel_size,
                        field_name=f"Kernel size. Layer: {layer.name}",
                        validate_failed=validate_failed,
                        validators=[
                            NoDashes(), Number(type_="int"),
                            DataRequiredIfDataInOtherFieldInValues(
                                other_field_value=layer.type_,
                                possible_values=["Conv2D"],
                            ),
                        ],
                    )
                    layer.kernel_size = to_number_or_None(layer.kernel_size, type_="int")

                    # pool size field
                    layer.pool_size, validate_failed = validate_field(
                        field_value=request.form.get(f"pool_size_{layer.id}"),
                        previous_value=layer.pool_size,
                        field_name=f"Pool size. Layer: {layer.name}",
                        validate_failed=validate_failed,
                        validators=[
                            NoDashes(), Number(type_="int"),
                            DataRequiredIfDataInOtherFieldInValues(
                                other_field_value=layer.type_,
                                possible_values=["MaxPooling2D"],
                            ),
                        ],
                    )
                    layer.pool_size = to_number_or_None(layer.pool_size, type_="int")

            if validate_failed:
                flash("Changes were not saved.", category="warning")
                return redirect(url_for("playground.main_page", neural_network_name=neural_network.name))

            db.session.commit()

        if create_layer_form.submitted():
            deep_learning_funcs.create_default_layer(neural_network)

        if train_form.submitted():
            show_results = True

            model, history = deep_learning_funcs.train_neural_network(neural_network)
            metrics_history = {
                "epochs": list(range(1, len(history.history["accuracy"])+1)),
                "accuracy": history.history["accuracy"],
                "val_accuracy": history.history["val_accuracy"],
                "loss": history.history["loss"],
                "val_loss": history.history["val_loss"],
            }

            layers_images = deep_learning_funcs.get_intermediate_representations(model, neural_network)

            # encode features images to send them to HTML
            layers_images_base64 = {}
            for layer_id, layer_images in layers_images.items():
                layers_images_base64[layer_id] = []
                for feature_img in layer_images:
                    img = Image.fromarray(feature_img.astype("uint8"))

                    file_object = io.BytesIO()
                    img.save(file_object, "JPEG")

                    file_object.seek(0)
                    img_base64 = base64.b64encode(file_object.getvalue()).decode("ascii")
                    mime = "image/jpeg"
                    uri = f"data:{mime};base64,{img_base64}"

                    layers_images_base64[layer_id].append(uri)

    return render_template(
        "playground/main.html", select_nn_form=select_nn_form,
        nn_parameters_form=nn_parameters_form,
        layers_parameters_form=layers_parameters_form, create_layer_form=create_layer_form,
        train_form=train_form, neural_networks=neural_networks,
        neural_network=neural_network, metrics_history=metrics_history,
        layers_images=layers_images_base64,
        show_results=show_results, optimizers=deep_learning_funcs.OPTIMIZERS,
        loss_functions=deep_learning_funcs.LOSS_FUNCTIONS,
        datasets_names=deep_learning_funcs.DATASETS_NAMES,
        layer_types=deep_learning_funcs.LAYER_TYPES,
        activation_functions=deep_learning_funcs.ACTIVATION_FUNCTIONS,
        len=len,
    )
