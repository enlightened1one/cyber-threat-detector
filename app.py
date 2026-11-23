"""Dataset-based cyber threat detection MVP."""
from __future__ import annotations

import io
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, abort, flash, redirect, render_template, request, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
db = SQLAlchemy()
login_manager = LoginManager()
csrf = CSRFProtect()


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="analyst")
    is_active = db.Column(db.Boolean, nullable=False, default=True)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def get_id(self) -> str:
        return str(self.id)


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    storage_name = db.Column(db.String(80), unique=True, nullable=False)
    checksum = db.Column(db.String(64), nullable=False)
    records = db.Column(db.Integer, nullable=False)
    uploaded_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    uploader_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    status = db.Column(db.String(20), nullable=False, default="ready")


class ModelVersion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("dataset.id"), nullable=False)
    artifact = db.Column(db.String(255), nullable=False)
    metrics_json = db.Column(db.Text, nullable=False)
    active = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    creator_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)


class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source_ip = db.Column(db.String(45))
    destination_ip = db.Column(db.String(45))
    protocol = db.Column(db.String(40))
    result = db.Column(db.String(30), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey("model_version.id"), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    review_status = db.Column(db.String(30), nullable=False, default="New")


class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    detection_id = db.Column(db.Integer, db.ForeignKey("detection.id"), nullable=False, unique=True)
    severity = db.Column(db.String(20), nullable=False, default="medium")
    acknowledged_at = db.Column(db.DateTime)
    acknowledged_by = db.Column(db.Integer, db.ForeignKey("user.id"))
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    detection = db.relationship("Detection")


class AnalystNote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    detection_id = db.Column(db.Integer, db.ForeignKey("detection.id"), nullable=False)
    author_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    content = db.Column(db.String(2000), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    author = db.relationship("User")


class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    actor_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    action = db.Column(db.String(100), nullable=False)
    target = db.Column(db.String(150))
    outcome = db.Column(db.String(30), nullable=False, default="success")
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


@login_manager.user_loader
def load_user(user_id: str):
    return db.session.get(User, int(user_id))


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
        SQLALCHEMY_DATABASE_URI=os.environ.get("DATABASE_URL", f"sqlite:///{ROOT / 'instance' / 'threats.db'}"),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        MAX_CONTENT_LENGTH=25 * 1024 * 1024,
        UPLOAD_DIR=Path(os.environ.get("UPLOAD_DIR", ROOT / "instance" / "uploads")),
        MODEL_DIR=Path(os.environ.get("MODEL_DIR", ROOT / "instance" / "models")),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE", "0") == "1",
    )
    if test_config:
        app.config.update(test_config)
    app.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    app.config["MODEL_DIR"].mkdir(parents=True, exist_ok=True)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "login"
    csrf.init_app(app)
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username=os.getenv("ADMIN_USERNAME", "admin")).first():
            admin = User(username=os.getenv("ADMIN_USERNAME", "admin"), role="administrator")
            initial_password = os.getenv("ADMIN_PASSWORD") or secrets.token_urlsafe(24)
            admin.set_password(initial_password)
            db.session.add(admin)
            db.session.commit()
            if not os.getenv("ADMIN_PASSWORD"):
                print(f"Initial administrator password (save it now): {initial_password}")
    @app.context_processor
    def template_helpers():
        unread_alerts = Alert.query.filter_by(acknowledged_at=None).count() if current_user.is_authenticated else 0
        return {"now": lambda: datetime.now(timezone.utc), "from_json": json.loads, "unread_alerts": unread_alerts}
    app.jinja_env.filters["from_json"] = json.loads
    register_routes(app)
    return app


def audit(action, target=None, outcome="success"):
    db.session.add(AuditLog(actor_id=current_user.id if current_user.is_authenticated else None,
                            action=action, target=str(target or ""), outcome=outcome))


def role_required(*roles):
    def decorate(fn):
        from functools import wraps
        @wraps(fn)
        @login_required
        def wrapped(*args, **kwargs):
            if current_user.role not in roles:
                abort(403)
            return fn(*args, **kwargs)
        return wrapped
    return decorate


def validate_frame(frame: pd.DataFrame, require_label=False):
    if frame.empty:
        raise ValueError("The CSV has no records.")
    if len(frame) > 250_000:
        raise ValueError("The CSV exceeds the 250,000 row limit.")
    frame.columns = [str(c).strip() for c in frame.columns]
    if len(set(frame.columns)) != len(frame.columns) or any(not c for c in frame.columns):
        raise ValueError("Column names must be non-empty and unique.")
    label = next((c for c in frame.columns if c.lower() in {"label", "class", "target", "attack", "prediction"}), None)
    if require_label and not label:
        raise ValueError("Training needs a label column named label, class, target or attack.")
    if frame.shape[1] < 2:
        raise ValueError("Include at least one feature column and a label column.")
    if require_label and frame[label].nunique(dropna=True) < 2:
        raise ValueError("Training requires at least two distinct classes.")
    return label


def register_routes(app):
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            user = User.query.filter_by(username=request.form.get("username", "").strip()).first()
            if user and user.is_active and check_password_hash(user.password_hash, request.form.get("password", "")):
                login_user(user, remember=False)
                audit("login", user.username)
                db.session.commit()
                return redirect(url_for("dashboard"))
            audit("login_failed", request.form.get("username", ""), "denied")
            db.session.commit()
            flash("Login failed. Check your credentials and try again.", "error")
        return render_template("login.html")

    @app.post("/logout")
    @login_required
    def logout():
        audit("logout", current_user.username)
        db.session.commit()
        logout_user()
        return redirect(url_for("login"))

    @app.get("/")
    @login_required
    def dashboard():
        total = Detection.query.count()
        malicious = Detection.query.filter(Detection.result != "Normal").count()
        return render_template("dashboard.html", total=total, malicious=malicious,
            datasets=Dataset.query.order_by(Dataset.uploaded_at.desc()).limit(5).all(),
            detections=Detection.query.order_by(Detection.created_at.desc()).limit(8).all(),
            active=ModelVersion.query.filter_by(active=True).first())

    @app.get("/datasets")
    @login_required
    def datasets():
        return render_template("datasets.html", datasets=Dataset.query.order_by(Dataset.uploaded_at.desc()).all())

    @app.route("/datasets/upload", methods=["GET", "POST"])
    @role_required("administrator", "analyst")
    def upload_dataset():
        if request.method == "POST":
            uploaded = request.files.get("file")
            if not uploaded or not uploaded.filename or Path(uploaded.filename).suffix.lower() != ".csv":
                flash("Choose a CSV file.", "error")
                return redirect(url_for("upload_dataset"))
            raw = uploaded.read()
            import hashlib
            digest = hashlib.sha256(raw).hexdigest()
            if Dataset.query.filter_by(checksum=digest).first():
                flash("This dataset has already been uploaded.", "error")
                return redirect(url_for("datasets"))
            try:
                frame = pd.read_csv(io.BytesIO(raw), nrows=250_001)
                validate_frame(frame)
            except Exception as exc:
                flash(f"CSV validation failed: {exc}", "error")
                return redirect(url_for("upload_dataset"))
            safe_name = secure_filename(uploaded.filename) or "dataset.csv"
            stored = f"{secrets.token_hex(16)}.csv"
            (app.config["UPLOAD_DIR"] / stored).write_bytes(raw)
            item = Dataset(name=request.form.get("name", "").strip() or safe_name,
                filename=safe_name[:255], storage_name=stored, checksum=digest, records=len(frame), uploader_id=current_user.id)
            db.session.add(item)
            db.session.flush()
            audit("dataset_uploaded", f"dataset:{item.id}")
            db.session.commit()
            flash(f"Uploaded {len(frame):,} records. Preview the columns before training.", "success")
            return redirect(url_for("dataset_detail", dataset_id=item.id))
        return render_template("upload.html")

    @app.get("/datasets/<int:dataset_id>")
    @login_required
    def dataset_detail(dataset_id):
        item = db.get_or_404(Dataset, dataset_id)
        path = app.config["UPLOAD_DIR"] / item.storage_name
        if not path.is_file():
            abort(404)
        frame = pd.read_csv(path, nrows=8)
        label = next((c for c in frame.columns if c.lower() in {"label", "class", "target", "attack", "prediction"}), None)
        return render_template("dataset_detail.html", item=item, columns=list(frame.columns), rows=frame.fillna("").to_dict("records"), label=label)

    @app.post("/datasets/<int:dataset_id>/train")
    @role_required("administrator", "analyst")
    def train(dataset_id):
        item = db.get_or_404(Dataset, dataset_id)
        path = app.config["UPLOAD_DIR"] / item.storage_name
        if not path:
            abort(404)
        try:
            frame = pd.read_csv(path)
            label = validate_frame(frame, require_label=True)
            y = frame[label].astype(str)
            X = frame.drop(columns=[label])
            if len(frame) < 10:
                raise ValueError("At least 10 records are required to create a held-out evaluation split.")
            stratify = y if y.value_counts().min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
            numeric = X.select_dtypes(include="number").columns.tolist()
            categorical = [c for c in X.columns if c not in numeric]
            transformers = []
            if numeric:
                transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
            if categorical:
                transformers.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical))
            pipeline = Pipeline([("features", ColumnTransformer(transformers)),
                                 ("classifier", RandomForestClassifier(n_estimators=160, class_weight="balanced", random_state=42, n_jobs=-1))])
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            metrics = {"accuracy": round(float(accuracy_score(y_test, pred)), 4),
                "classes": list(map(str, pipeline.classes_)),
                "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, pred, labels=pipeline.classes_).tolist(),
                "test_records": len(y_test), "features": list(X.columns), "label": label}
            version = (ModelVersion.query.count() + 1)
            artifact = f"model-{version}-{secrets.token_hex(4)}.joblib"
            joblib.dump({"pipeline": pipeline, "features": list(X.columns), "label": label}, app.config["MODEL_DIR"] / artifact)
            model = ModelVersion(dataset_id=item.id, artifact=artifact, metrics_json=json.dumps(metrics), creator_id=current_user.id)
            db.session.add(model)
            db.session.flush()
            audit("model_trained", f"model:{model.id}")
            db.session.commit()
            flash(f"Training complete. Held-out accuracy: {metrics['accuracy']:.1%}. Activate the model after review.", "success")
            return redirect(url_for("models"))
        except Exception as exc:
            flash(f"Training failed: {exc}", "error")
            return redirect(url_for("dataset_detail", dataset_id=item.id))

    @app.get("/models")
    @login_required
    def models():
        return render_template("models.html", models=ModelVersion.query.order_by(ModelVersion.created_at.desc()).all())

    @app.post("/models/<int:model_id>/activate")
    @role_required("administrator")
    def activate(model_id):
        model = db.get_or_404(ModelVersion, model_id)
        ModelVersion.query.update({ModelVersion.active: False})
        model.active = True
        audit("model_activated", f"model:{model.id}")
        db.session.commit()
        flash("Model activated. New predictions will use this version.", "success")
        return redirect(url_for("models"))

    @app.route("/predict", methods=["GET", "POST"])
    @role_required("administrator", "analyst")
    def predict():
        model = ModelVersion.query.filter_by(active=True).first()
        if request.method == "POST":
            uploaded = request.files.get("file")
            if not model:
                flash("Train and activate a model before predicting.", "error")
            elif not uploaded or Path(uploaded.filename or "").suffix.lower() != ".csv":
                flash("Choose an input CSV file.", "error")
            else:
                try:
                    payload = joblib.load(app.config["MODEL_DIR"] / model.artifact)
                    frame = pd.read_csv(uploaded, nrows=250_001)
                    if len(frame) > 250_000:
                        raise ValueError("The CSV exceeds the 250,000 row limit.")
                    missing = set(payload["features"]) - set(frame.columns)
                    if missing:
                        raise ValueError(f"Missing model features: {', '.join(sorted(missing))}")
                    X = frame[payload["features"]]
                    output = payload["pipeline"].predict(X)
                    batch = []
                    for index, value in enumerate(output):
                        row = X.iloc[index]
                        def field(key):
                            found = next((c for c in row.index if c.lower() == key), None)
                            return str(row[found])[:45] if found and pd.notna(row[found]) else None
                        result = str(value)
                        batch.append(Detection(source_ip=field("source_ip"), destination_ip=field("destination_ip"),
                            protocol=(str(row[next(c for c in row.index if c.lower() == "protocol")])[:40]
                                if any(c.lower() == "protocol" for c in row.index) else None), result=result, model_id=model.id))
                    db.session.add_all(batch)
                    db.session.flush()
                    for detection in batch:
                        if detection.result.strip().lower() not in {"normal", "benign", "0"}:
                            db.session.add(Alert(detection_id=detection.id, severity="medium"))
                    audit("batch_prediction", f"model:{model.id};rows:{len(batch)}")
                    db.session.commit()
                    malicious = sum(d.result.lower() not in {"normal", "benign", "0"} for d in batch)
                    flash(f"Classified {len(batch):,} records; {malicious:,} marked malicious. No live network capture is enabled.", "success")
                    return redirect(url_for("threats"))
                except Exception as exc:
                    db.session.rollback()
                    flash(f"Prediction failed: {exc}", "error")
        return render_template("predict.html", model=model)

    @app.get("/threats")
    @login_required
    def threats():
        query = Detection.query
        term = request.args.get("q", "").strip()
        status = request.args.get("status", "")
        classification = request.args.get("classification", "")
        if term:
            query = query.filter(db.or_(Detection.source_ip.contains(term), Detection.destination_ip.contains(term), Detection.protocol.contains(term)))
        if status in {"New", "Reviewing", "Confirmed", "False Positive"}:
            query = query.filter_by(review_status=status)
        if classification:
            query = query.filter(Detection.result.ilike(f"%{classification}%"))
        detections = query.order_by(Detection.created_at.desc()).limit(500).all()
        notes = AnalystNote.query.filter(AnalystNote.detection_id.in_([d.id for d in detections])).order_by(AnalystNote.created_at.asc()).all() if detections else []
        notes_by_detection = {}
        for note in notes:
            notes_by_detection.setdefault(note.detection_id, []).append(note)
        return render_template("threats.html", detections=detections, notes_by_detection=notes_by_detection,
                               filters={"q": term, "status": status, "classification": classification})

    @app.post("/threats/<int:detection_id>/notes")
    @role_required("administrator", "analyst")
    def add_note(detection_id):
        db.get_or_404(Detection, detection_id)
        content = request.form.get("content", "").strip()
        if not content or len(content) > 2000:
            flash("Investigation notes must contain 1–2,000 characters.", "error")
            return redirect(url_for("threats"))
        note = AnalystNote(detection_id=detection_id, author_id=current_user.id, content=content)
        db.session.add(note)
        audit("investigation_note_added", f"detection:{detection_id}")
        db.session.commit()
        return redirect(url_for("threats"))

    @app.post("/threats/<int:detection_id>/status")
    @role_required("administrator", "analyst")
    def update_status(detection_id):
        item = db.get_or_404(Detection, detection_id)
        status = request.form.get("status")
        if status not in {"New", "Reviewing", "Confirmed", "False Positive"}:
            abort(400)
        item.review_status = status
        audit("threat_review_updated", f"detection:{item.id}")
        db.session.commit()
        return redirect(url_for("threats"))

    @app.get("/reports.csv")
    @role_required("administrator", "analyst", "viewer")
    def report_csv():
        rows = Detection.query.order_by(Detection.created_at.desc()).all()
        from flask import Response
        import csv
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "detected_at", "source_ip", "destination_ip", "protocol", "prediction", "review_status", "model_version"])
        for d in rows:
            writer.writerow([d.id, d.created_at.isoformat(), d.source_ip, d.destination_ip, d.protocol, d.result, d.review_status, d.model_id])
        return Response(output.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=threat-report.csv"})

    @app.get("/audit")
    @role_required("administrator")
    def audit_view():
        return render_template("audit.html", entries=AuditLog.query.order_by(AuditLog.created_at.desc()).limit(300).all())

    @app.get("/alerts")
    @login_required
    def alerts():
        items = Alert.query.join(Detection).order_by(Alert.acknowledged_at.is_(None).desc(), Alert.created_at.desc()).limit(500).all()
        return render_template("alerts.html", alerts=items)

    @app.post("/alerts/<int:alert_id>/acknowledge")
    @role_required("administrator", "analyst")
    def acknowledge_alert(alert_id):
        alert = db.get_or_404(Alert, alert_id)
        if alert.acknowledged_at is None:
            alert.acknowledged_at = datetime.now(timezone.utc)
            alert.acknowledged_by = current_user.id
            audit("alert_acknowledged", f"alert:{alert.id}")
            db.session.commit()
        return redirect(url_for("alerts"))

    @app.route("/users", methods=["GET", "POST"])
    @role_required("administrator")
    def users():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            role = request.form.get("role", "")
            if len(username) < 3 or len(username) > 80 or not username.replace("_", "").replace("-", "").isalnum():
                flash("Use a username of 3–80 letters, numbers, hyphens or underscores.", "error")
            elif len(password) < 12:
                flash("Use a temporary password with at least 12 characters.", "error")
            elif role not in {"administrator", "analyst", "viewer"}:
                flash("Choose a valid role.", "error")
            elif User.query.filter_by(username=username).first():
                flash("That username already exists.", "error")
            else:
                user = User(username=username, role=role)
                user.set_password(password)
                db.session.add(user)
                db.session.flush()
                audit("user_created", f"user:{user.id};role:{role}")
                db.session.commit()
                flash(f"Created {role} account for {username}.", "success")
                return redirect(url_for("users"))
        return render_template("users.html", users=User.query.order_by(User.username).all())

    @app.post("/users/<int:user_id>/toggle")
    @role_required("administrator")
    def toggle_user(user_id):
        user = db.get_or_404(User, user_id)
        if user.id == current_user.id:
            flash("You cannot deactivate your own account.", "error")
        else:
            user.is_active = not user.is_active
            audit("user_status_changed", f"user:{user.id};active:{user.is_active}")
            db.session.commit()
            flash(f"Account {user.username} is now {'active' if user.is_active else 'inactive'}.", "success")
        return redirect(url_for("users"))


app = create_app()

if __name__ == "__main__":
    app.run(debug=False)
