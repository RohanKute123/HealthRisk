
from flask import Flask

from config import Config
from extensions import db
from models import Assessment


def initialize_database():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)

    with app.app_context():
        db.create_all()

        print("Database tables created successfully.")
        print("Table: assessments")


if __name__ == "__main__":
    initialize_database()
