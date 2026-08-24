from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_config_loads_and_mlflow_is_enabled():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    assert config["mlflow"]["enabled"] is True
    assert config["mlflow"]["experiment_name"] == "FraudX-Fraud-Detection"
    assert config["evaluation"]["primary_metric"] == "pr_auc"
    assert config["features"]["velocity"]["enabled"] is True


def test_required_project_files_exist():
    required = [
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
        "train.py",
        "api/main.py",
        "api/dependencies.py",
        "api/mongodb.py",
        "src/models/ensemble.py",
        "src/monitoring/drift.py",
        "tests/test_monitoring.py",
    ]

    for relative_path in required:
        assert (ROOT / relative_path).is_file(), relative_path


def test_compose_exposes_fraudx_api_on_8001():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert '"8001:8000"' in compose
    assert '"27017:27017"' in compose
    assert '"5000:5000"' in compose
