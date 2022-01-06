from pathlib import Path
STATIC_DIR = (str(Path(__file__).parent) + "/static").replace("\\", "/")

from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

app = Flask(__name__)
db = SQLAlchemy()
flask_bcrypt = Bcrypt()

@app.errorhandler(404)
def not_found(error):
    return render_template("error/404.html"), 404

def initialize_engines():
    """
    Initializes the SQLAlchemy and Bcrypt engines.
    """
    db.app = app
    db.init_app(app)
    flask_bcrypt.init_app(app)

def register_blueprints():
    """
    Registers the blueprints on the application.
    """    
    from app.main.controllers import main_bp
    from app.playground.controllers import playground_bp
    from app.authentication.controllers import authentication_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(playground_bp)
    app.register_blueprint(authentication_bp)
