
from datetime import datetime, timezone

from extensions import db


class Assessment(db.Model):
    __tablename__ = "assessments"

    id = db.Column(db.Integer, primary_key=True)

    pregnancies = db.Column(db.Integer, nullable=False)
    glucose = db.Column(db.Float, nullable=False)
    blood_pressure = db.Column(db.Float, nullable=False)
    skin_thickness = db.Column(db.Float, nullable=False)
    insulin = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)

    diabetes_pedigree_function = db.Column(
        db.Float, nullable=False
    )

    age = db.Column(db.Integer, nullable=False)

    # 0 or 1: the trained model's predicted class
    prediction = db.Column(db.Integer, nullable=False)

    # Probability estimated by the model, from 0 to 100
    risk_probability = db.Column(db.Float, nullable=True)

    model_version = db.Column(
        db.String(50), nullable=False, default="diabetes-v1"
    )

    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True
    )

    def to_dict(self):
        return {
            "id": self.id,
            "prediction": self.prediction,
            "risk_probability": self.risk_probability,
            "model_version": self.model_version,
            "created_at": (
                self.created_at.isoformat()
                if self.created_at else None
            )
        }
