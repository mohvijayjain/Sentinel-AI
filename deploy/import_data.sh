#!/usr/bin/env bash
# Run ONCE on the EC2 server, from the repository root, after .env is filled in
# and sentinel-transfer/ has been copied to the home directory:
#     bash deploy/import_data.sh [~/sentinel-transfer]
#
# Restores the laptop's data into the fresh server stack, then starts
# everything. Order matters: Postgres is restored BEFORE MLflow / the API
# start, so MLflow does not create its own empty tables first.

set -euo pipefail

transfer="${1:-$HOME/sentinel-transfer}"
compose="docker compose -f docker-compose.yml -f deploy/docker-compose.aws.yml"

for f in sentinel.dump mlflow_artifacts.tgz data.tgz; do
    [ -f "$transfer/$f" ] || { echo "Missing $transfer/$f"; exit 1; }
done
[ -f .env ] || { echo "Missing .env (cp deploy/.env.aws.example .env)"; exit 1; }

echo "==> Data files (reference, processed, frozen test, April raw, Chroma)"
tar -xzf "$transfer/data.tgz" -C .

echo "==> Building images (the API image includes data/, so after extraction)"
$compose build

echo "==> Starting PostgreSQL only"
$compose up -d postgres
until [ "$(docker inspect -f '{{.State.Health.Status}}' sentinel-postgres)" = "healthy" ]; do
    sleep 2
done

echo "==> Restoring the database"
docker cp "$transfer/sentinel.dump" sentinel-postgres:/tmp/sentinel.dump
# --no-owner: objects belong to this server's POSTGRES_USER. pg_restore may
# print harmless warnings (e.g. about the plpgsql extension); the row counts
# below are the real check.
docker exec sentinel-postgres sh -c \
    'pg_restore --no-owner --no-privileges --clean --if-exists -U "$POSTGRES_USER" -d "$POSTGRES_DB" /tmp/sentinel.dump' \
    || echo "(pg_restore reported warnings, see above)"
docker exec sentinel-postgres rm /tmp/sentinel.dump

docker exec sentinel-postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
    SELECT (SELECT count(*) FROM monitoring_runs)   AS monitoring_runs,
           (SELECT count(*) FROM retraining_events) AS retraining_events,
           (SELECT count(*) FROM prediction_logs)   AS prediction_logs,
           (SELECT count(*) FROM drift_scores)      AS drift_scores;"'

echo "==> Restoring MLflow artifacts"
$compose run --rm -T --no-deps --entrypoint tar mlflow \
    xzf - -C /mlflow/artifacts < "$transfer/mlflow_artifacts.tgz"

echo "==> Starting everything"
$compose up -d

echo
echo "Done. In a minute or two check:  curl https://\$SITE_ADDRESS/health"
echo "The transfer folder holds a full DB dump: delete it when you are happy:"
echo "    rm -rf $transfer"
