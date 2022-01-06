from app import db, flask_bcrypt

class NeuralNetwork(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(length=30), nullable=False, unique=True)
    layers = db.relationship("Layer", backref="neural_network", lazy=True)
    loss = db.Column(db.String(length=30), nullable=False)
    optimizer = db.Column(db.String(length=30), nullable=False)
    epochs = db.Column(db.Integer())
    learning_rate = db.Column(db.Float())
    dataset_name = db.Column(db.String(length=30), nullable=False)
    creator_id = db.Column(db.Integer(), db.ForeignKey("user.id"))

    @property
    def sorted_layers(self):
        return sorted(self.layers)

class Layer(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(length=30), nullable=False)
    type_ = db.Column(db.String(length=30), nullable=False)
    neural_network_id = db.Column(
        db.Integer(), db.ForeignKey("neural_network.id"),
    )
    index = db.Column(db.Integer()) # index within neural net layers
    units = db.Column(db.Integer(), nullable=True)
    filters = db.Column(db.Integer(), nullable=True)
    activation_function = db.Column(db.String(20), nullable=True)
    kernel_size = db.Column(db.Integer(), nullable=True)
    pool_size = db.Column(db.Integer(), nullable=True)

    # sort list of Layer objects by index property
    def __lt__(self, other):
        return self.index < other.index
