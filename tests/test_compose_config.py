"""
docker-compose.yml carries no literal Postgres credentials: they come
from .env via ${VAR} references, documented in .env.example, and .env is
gitignored. Static checks only (no Docker needed).
"""

import os
import re

import pytest
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

COMPOSE = os.path.join(REPO_ROOT, "docker-compose.yml")
ENV_EXAMPLE = os.path.join(REPO_ROOT, ".env.example")
GITIGNORE = os.path.join(REPO_ROOT, ".gitignore")

REQUIRED_VARS = ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB",
                 "POSTGRES_URL"]

VAR_REF = re.compile(r"^\$\{[A-Z_]+(:-[^}]*)?\}$")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load(_read(COMPOSE))


def test_old_literal_password_gone():

    assert "sentinel123" not in _read(COMPOSE)


def test_postgres_service_credentials_are_var_references(compose):

    env = compose["services"]["postgres"]["environment"]

    for key in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
        assert VAR_REF.match(env[key]), (key, env[key])
        assert env[key].startswith("${" + key)


def test_password_default_is_a_local_placeholder(compose):

    password = compose["services"]["postgres"]["environment"]["POSTGRES_PASSWORD"]

    assert password == "${POSTGRES_PASSWORD:-changeme-local}"


def test_no_literal_password_in_any_database_url():

    urls = re.findall(r"postgres(?:ql)?://[^\s\"']+", _read(COMPOSE))

    assert urls, "expected the MLflow backend store URL"

    for url in urls:
        # user and password must each be a whole ${...} reference
        # (the default syntax ":-" means a plain split on ":" won't do)
        userinfo = re.match(
            r"postgres(?:ql)?://(\$\{[^}]*\}):(\$\{[^}]*\})@", url
        )

        assert userinfo, f"literal credentials in {url}"
        assert userinfo.group(1).startswith("${POSTGRES_USER"), url
        assert userinfo.group(2).startswith("${POSTGRES_PASSWORD"), url


def test_mlflow_backend_url_built_from_vars(compose):

    # Passed via MLFLOW_BACKEND_STORE_URI (mlflow server's envvar for
    # --backend-store-uri), not on the command line: see 1F tests below
    uri = compose["services"]["mlflow"]["environment"]["MLFLOW_BACKEND_STORE_URI"]

    assert uri == (
        "postgresql://${POSTGRES_USER:-sentinel}:"
        "${POSTGRES_PASSWORD:-changeme-local}@postgres:5432/"
        "${POSTGRES_DB:-sentinel_db}"
    )


def test_no_credentials_in_any_service_command(compose):
    """Command-line args show in the process list and docker inspect."""

    for name, service in compose["services"].items():
        command = service.get("command") or []
        if isinstance(command, str):
            command = command.split()

        for arg in command:
            assert "://" not in str(arg), (name, arg)
            assert "POSTGRES_PASSWORD" not in str(arg), (name, arg)

    assert "--backend-store-uri" not in compose["services"]["mlflow"]["command"]


def test_obsolete_version_key_removed(compose):

    assert "version" not in compose


def test_services_and_wiring_intact(compose):

    services = compose["services"]

    assert set(services) == {"sentinel-api", "postgres", "mlflow"}
    assert services["mlflow"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert services["sentinel-api"]["depends_on"]["mlflow"]["condition"] == "service_healthy"
    assert services["mlflow"]["healthcheck"]["test"][0] == "CMD"
    assert "mlflow_artifacts:/mlflow/artifacts" in services["mlflow"]["volumes"]
    assert "postgres_data:/var/lib/postgresql/data" in services["postgres"]["volumes"]
    assert set(compose["volumes"]) == {"postgres_data", "mlflow_artifacts"}
    for service in services.values():
        assert service["networks"] == ["sentinel-network"]


def test_healthcheck_uses_container_env(compose):

    test = compose["services"]["postgres"]["healthcheck"]["test"]

    assert "$${POSTGRES_USER}" in test[1]
    assert "$${POSTGRES_DB}" in test[1]


def test_env_example_documents_required_vars():

    text = _read(ENV_EXAMPLE)
    keys = re.findall(r"^([A-Z_]+)=", text, flags=re.M)

    for var in REQUIRED_VARS:
        assert var in keys, var

    # Placeholders only, never the old real password
    assert "sentinel123" not in text


def test_env_is_gitignored_but_example_is_not():

    patterns = [line.strip() for line in _read(GITIGNORE).splitlines()]

    assert ".env" in patterns
    assert ".env.example" not in patterns
    assert not any(p in (".env*", "*.example") for p in patterns)
