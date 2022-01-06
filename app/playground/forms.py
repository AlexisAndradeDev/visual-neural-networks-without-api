from flask_wtf import FlaskForm
from wtforms import SubmitField

# Note: The fields objects of the forms must have different names.

class SelectNeuralNetworkForm(FlaskForm):
    selectNeuralNetwork = SubmitField(label="Select")

    def submitted(self):
        return self.selectNeuralNetwork.data and self.validate()

class NeuralNetworkParametersForm(FlaskForm):
    submitNNParameters = SubmitField(label="Save")

    def submitted(self):
        return self.submitNNParameters.data and self.validate()

class LayersParametersForm(FlaskForm):
    submitLayersParameters = SubmitField(label="Save")

    def submitted(self):
        return self.submitLayersParameters.data and self.validate()

class CreateLayerForm(FlaskForm):
    submitCreateLayer = SubmitField(label="Create new layer")

    def submitted(self):
        return self.submitCreateLayer.data and self.validate()

class TrainForm(FlaskForm):
    submitTrain = SubmitField(label="Run")

    def submitted(self):
        return self.submitTrain.data and self.validate()
