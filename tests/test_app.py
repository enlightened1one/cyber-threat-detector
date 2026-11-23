"""End-to-end checks for the main dataset-to-detection workflow."""
import io
import os
import tempfile
import unittest
from pathlib import Path

os.environ["ADMIN_USERNAME"] = "test-admin"
os.environ["ADMIN_PASSWORD"] = "Unit-Test-Password-9021!"
os.environ["SECRET_KEY"] = "test-only-secret-key"
os.environ["DATABASE_URL"] = "sqlite://"

from app import Alert, AnalystNote, AuditLog, Dataset, Detection, ModelVersion, User, create_app, db  # noqa: E402


def traffic_csv(rows=40, include_label=True):
    lines = ["src_port,protocol,source_ip" + (",label" if include_label else "")]
    for i in range(rows):
        fields = [str(4000 + i), "tcp" if i % 2 else "udp", f"10.0.0.{i + 1}"]
        if include_label:
            fields.append("Normal" if i % 2 == 0 else "Malicious")
        lines.append(",".join(fields))
    return "\n".join(lines).encode()


class ThreatAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.app = create_app({
            "TESTING": True,
            "WTF_CSRF_ENABLED": False,
            "SECRET_KEY": "test-secret",
            "SQLALCHEMY_DATABASE_URI": "sqlite://",
            "UPLOAD_DIR": root / "uploads",
            "MODEL_DIR": root / "models",
            "MAX_CONTENT_LENGTH": 1024 * 1024,
        })
        self.client = self.app.test_client()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()
        self.temp.cleanup()

    def login(self, username="test-admin", password="Unit-Test-Password-9021!"):
        return self.client.post("/login", data={"username": username, "password": password}, follow_redirects=True)

    def create_user(self, username, role):
        with self.app.app_context():
            user = User(username=username, role=role)
            user.set_password("Long-Test-Password-9021!")
            db.session.add(user)
            db.session.commit()

    def upload(self, payload=None, filename="traffic.csv"):
        return self.client.post("/datasets/upload", data={
            "name": "Unit test traffic",
            "file": (io.BytesIO(payload or traffic_csv()), filename),
        }, content_type="multipart/form-data", follow_redirects=True)

    def test_health_and_authentication(self):
        self.assertEqual(self.client.get("/health").json, {"status": "ok"})
        self.assertEqual(self.client.get("/").status_code, 302)
        response = self.client.post("/login", data={"username": "test-admin", "password": "wrong"}, follow_redirects=True)
        self.assertIn(b"Login failed", response.data)
        self.assertTrue(self.login().request.path == "/")

    def test_invalid_and_duplicate_datasets_are_rejected(self):
        self.login()
        response = self.upload(b"only_one_column\nvalue\n")
        self.assertIn(b"validation failed", response.data.lower())
        self.upload()
        response = self.upload()
        self.assertIn(b"already been uploaded", response.data)
        with self.app.app_context():
            self.assertEqual(Dataset.query.count(), 1)

    def test_training_activation_batch_prediction_review_and_report(self):
        self.login()
        self.upload()
        with self.app.app_context():
            dataset = Dataset.query.one()
            dataset_id = dataset.id
        preview = self.client.get(f"/datasets/{dataset_id}")
        self.assertEqual(preview.status_code, 200)
        self.assertIn(b"Record preview", preview.data)

        response = self.client.post(f"/datasets/{dataset_id}/train", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Training complete", response.data)
        with self.app.app_context():
            model = ModelVersion.query.one()
            model_id = model.id
            metrics = model.metrics_json
            self.assertIn('"accuracy"', metrics)
            self.assertFalse(model.active)

        response = self.client.post(f"/models/{model_id}/activate", follow_redirects=True)
        self.assertIn(b"Model activated", response.data)
        prediction_csv = b"source_ip,src_port,protocol\n192.0.2.1,80,tcp\n192.0.2.2,53,udp\n"
        response = self.client.post("/predict", data={
            "file": (io.BytesIO(prediction_csv), "batch.csv"),
        }, content_type="multipart/form-data", follow_redirects=True)
        self.assertIn(b"Classified 2 records", response.data)
        with self.app.app_context():
            detections = Detection.query.order_by(Detection.id).all()
            self.assertEqual(len(detections), 2)
            detection_id = detections[0].id
            self.assertTrue(all(d.model_id == model_id for d in detections))

        response = self.client.post(f"/threats/{detection_id}/status", data={"status": "Confirmed"}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        with self.app.app_context():
            self.assertEqual(db.session.get(Detection, detection_id).review_status, "Confirmed")
            alert_count = Alert.query.count()
            self.assertGreaterEqual(alert_count, 1)
            alert_id = Alert.query.first().id
        notes = self.client.post(f"/threats/{detection_id}/notes", data={"content": "Validated source context."}, follow_redirects=True)
        self.assertEqual(notes.status_code, 200)
        with self.app.app_context():
            self.assertEqual(AnalystNote.query.filter_by(detection_id=detection_id).one().content, "Validated source context.")
        alert_page = self.client.get("/alerts")
        self.assertEqual(alert_page.status_code, 200)
        acknowledged = self.client.post(f"/alerts/{alert_id}/acknowledge", follow_redirects=True)
        self.assertEqual(acknowledged.status_code, 200)
        with self.app.app_context():
            self.assertIsNotNone(db.session.get(Alert, alert_id).acknowledged_at)
        report = self.client.get("/reports.csv")
        self.assertEqual(report.status_code, 200)
        self.assertIn(b"prediction", report.data)
        audit = self.client.get("/audit")
        self.assertEqual(audit.status_code, 200)
        with self.app.app_context():
            self.assertGreaterEqual(AuditLog.query.count(), 5)

    def test_viewer_can_read_but_cannot_mutate(self):
        self.create_user("reader", "viewer")
        self.login("reader", "Long-Test-Password-9021!")
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/reports.csv").status_code, 200)
        self.assertEqual(self.client.get("/audit").status_code, 403)
        self.assertEqual(self.client.get("/datasets/upload").status_code, 403)

    def test_prediction_rejects_missing_features_without_writing_rows(self):
        self.login()
        self.upload()
        with self.app.app_context():
            dataset_id = Dataset.query.one().id
        self.client.post(f"/datasets/{dataset_id}/train", follow_redirects=True)
        with self.app.app_context():
            model_id = ModelVersion.query.one().id
        self.client.post(f"/models/{model_id}/activate", follow_redirects=True)
        response = self.client.post("/predict", data={
            "file": (io.BytesIO(b"src_port\n80\n"), "incomplete.csv"),
        }, content_type="multipart/form-data", follow_redirects=True)
        self.assertIn(b"Missing model features", response.data)
        with self.app.app_context():
            self.assertEqual(Detection.query.count(), 0)


if __name__ == "__main__":
    unittest.main()
