# Run on the LAPTOP (Windows PowerShell), from the repository root, with the
# local Docker stack running:
#     .\deploy\export_local_data.ps1
#
# Writes sentinel-transfer\ (gitignored) with everything git does not carry:
#   sentinel.dump          PostgreSQL: monitoring runs, retraining events,
#                          prediction logs, drift scores, MLflow metadata
#   mlflow_artifacts.tgz   MLflow artifact volume
#   data.tgz               reference + processed data, frozen test, April raw
#                          (retraining's recent-data gate), current Chroma index
# Read-only for the local stack: nothing is modified or deleted there.

$ErrorActionPreference = "Stop"
$out = "sentinel-transfer"
New-Item -ItemType Directory -Force $out | Out-Null

Write-Host "==> PostgreSQL dump"
# Written inside the container: a PowerShell redirect would corrupt the binary dump
docker exec sentinel-postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc -f /tmp/sentinel.dump'
docker cp sentinel-postgres:/tmp/sentinel.dump "$out/sentinel.dump"
docker exec sentinel-postgres rm /tmp/sentinel.dump

Write-Host "==> MLflow artifacts"
docker run --rm -v sentinel-ai_mlflow_artifacts:/a -v "${PWD}/${out}:/b" alpine tar czf /b/mlflow_artifacts.tgz -C /a .

Write-Host "==> Data files"
tar -czf "$out/data.tgz" `
    data/reference `
    data/processed `
    data/frozen_test.parquet `
    data/raw/yellow_tripdata_2026-04.parquet `
    data/chroma

Get-ChildItem $out | Format-Table Name, @{n = "MB"; e = { [math]::Round($_.Length / 1MB, 1) } }
Write-Host "Done. Copy the folder to the server (see deploy/README.md, step 5)."
