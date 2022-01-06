from app import db, flask_bcrypt

class User(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    username = db.Column(db.String(length=30), unique=True, nullable=False)
    email = db.Column(db.String(length=50), nullable=False, unique=True)
    password_hash = db.Column(db.String(length=60), nullable=False)
    created_neural_networks = db.relationship(
        "NeuralNetwork", backref="creator_user", lazy=True,
    )

    @property
    def password(self):
        raise AttributeError("password: Write-only field.")
    
    @password.setter
    def password(self, plain_text_password):
        self.password_hash = flask_bcrypt.generate_password_hash(plain_text_password).decode("utf-8")

    def check_password(self, attempted_password):
        return flask_bcrypt.check_password_hash(self.password_hash, attempted_password)
    
    def __repr__(self):
        return f"<User {self.username}>"
