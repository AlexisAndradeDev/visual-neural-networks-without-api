import unittest, sys

from app import app, db, initialize_engines, register_blueprints
from app.config import config_by_name

def load_config(app):
    """Sets the configuration (dev, test, production).
    If a configuration name is not specified, the configuration will be 
    'production' by default."""
    if "-config" in sys.argv:
        config_name_index = sys.argv.index("-config") + 1
        assert config_name_index < len(sys.argv), "Specify a configuration name after -config. Example: -config production"
        config_name = sys.argv[config_name_index]
    else:
        config_name = "production" # default

    assert config_name in ["dev", "test", "production"], "-config name must be: dev, test or production."

    config = config_by_name[config_name]
    app.config.from_object(config)

def runserver():
    app.run()

def test():
    tests = unittest.TestLoader().discover("test", pattern="test*.py")
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if result.wasSuccessful():
        return 0
    return 1

if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] not in ["runserver", "test"]:
        raise Exception("Introduce a command: runserver, test.")
    else:
        load_config(app)
        initialize_engines()
        register_blueprints()
        app.app_context().push()

        # create database if it does not exist
        db.create_all()

        command = sys.argv[1]
        if command == "runserver":
            runserver()
        elif command == "test":
            test()
        else:
            raise Exception("Command does not exist.")
