# Sentinel — Cyber Threat Detection

A dataset-based cybersecurity analysis MVP built from the supplied software blueprint. It supports role-based accounts, private CSV imports, Random Forest model training and held-out evaluation, administrator-controlled model activation, batch inference, threat review, CSV reports and an audit trail.

**This application does not capture live network traffic.** It classifies records supplied by authorized users. Live sensors, packet/flow collection, streaming ingestion, email notifications and multi-organization hosting are outside this MVP.

## Run locally

Requires Python 3.10 or newer.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and replace `SECRET_KEY` and `ADMIN_PASSWORD` before the first run. If you skip `ADMIN_PASSWORD`, the app generates a random one-time initial password and prints it to the server console. The administrator account is created only when the database is first initialized; subsequent credential changes do not rewrite an existing account password. Keep `.env`, the `instance/` directory, uploaded datasets and model artifacts private. Never use the example values in a deployed environment.

```powershell
python app.py
```

Open http://127.0.0.1:5000 and sign in with the configured administrator credentials. For production serving on Windows, run `waitress-serve --call app:create_app`. Configure HTTPS, a strong secret, secure cookies (`COOKIE_SECURE=1`), database backups and private storage before deployment.

SQLite is the default for a local demo. Set `DATABASE_URL` to a MySQL SQLAlchemy URL to use MySQL, for example `mysql+pymysql://user:password@localhost:3306/threat_detector?charset=utf8mb4`. Create the database and least-privilege database user separately. This first version creates tables at startup; a migration system is a follow-on before schema changes in a deployed database.

## MVP workflow

1. Sign in as the administrator. Create analyst and viewer accounts in **User access**.
2. Upload a CSV in **Datasets**. CSV uploads are capped at 25 MB and 250,000 rows, checked for non-empty columns, deduplicated by SHA-256, and stored under a server-generated private filename.
3. Preview the dataset. For training, include a target column named `label`, `class`, `target` or `attack` and at least one feature column.
4. Train a Random Forest. The preprocessing pipeline is fitted only on the training split and handles missing numeric values and categorical encoding. The app records accuracy, per-class metrics, confusion matrix, feature names and test-set size. Research results in the blueprint are not hard-coded or assumed.
5. An administrator reviews the evaluation and activates the model. Earlier model versions are retained; activation switches inference to one version.
6. Upload feature-only CSV records in **Analyze traffic**. The app validates that all trained features exist, predicts the batch, stores results and exposes review status in **Threat log**.

Malicious predictions create in-app alerts with an acknowledgement state. Every alert currently receives a default medium triage level; it is not a calibrated risk score. Analysts can filter the threat log by IP/protocol, class and review state, and attach investigation notes to detections.

The first release supports whichever labels are present in a trained dataset; it does not invent attack categories. A deployment should establish whether its data is binary or multiclass and curate labels accordingly. The dashboard counts every non-Normal class as a threat, so label the benign class `Normal` (or `Benign`) for correct KPI semantics.

## Roles

| Role | Capabilities |
| --- | --- |
| Administrator | Manage accounts, upload/train, activate models, analyze traffic, review threats, export reports, inspect audit history |
| Security analyst | Upload/train, analyze traffic, review threats, export reports |
| Viewer | Read dashboards, datasets, model metrics and threat history |

Passwords use Werkzeug's password hashing. Browser forms use CSRF protection, session cookies are HTTP-only and SameSite=Lax, and role checks protect mutating operations. This MVP does not yet provide login throttling, password reset/MFA, immutable external audit storage, background training jobs, database migrations or a formal deployment hardening review.

## Data and model safety

- Uploaded CSVs and joblib model artifacts are outside static/public paths. Generated filenames prevent path traversal and content is treated as data.
- Joblib files are only loaded from the app-controlled model directory. Do not put untrusted artifacts there.
- Inference accepts CSV input only and caps input size/rows. It uses the saved preprocessing pipeline and rejects missing model features.
- Predictions are in-app records; there is no email/SMS channel or live sensor.
- Restrict access to the host, database, backups and `instance/` storage. Define retention and authorization before importing production traffic.

## Repository layout

```text
app.py              Flask application, data models, training and inference
templates/          Server-rendered application screens
static/app.css      Responsive interface styling
instance/           Private SQLite database, uploads and model artifacts (ignored)
```
