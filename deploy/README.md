# Deploy Sentinel-AI on AWS (one EC2 instance)

The existing Docker Compose stack (API, PostgreSQL, MLflow) runs on one EC2 instance, with **Caddy** in front providing automatic HTTPS. The Streamlit Cloud UI then talks to `https://<your-address>`.

Only ports 80 and 443 are public. PostgreSQL, MLflow and the API's own port are not reachable from the internet.

```
Streamlit Cloud ──HTTPS──▶ Caddy :443 ──▶ sentinel-api :8000 ──▶ postgres / mlflow
                           (EC2, only public entry point)
```

## 0. Before you start

- **AWS account on the Free plan, with protections in place:**
  - Turn on MFA for the root user.
  - Create a zero-spend budget: Billing → Budgets.
- **Region:** Asia Pacific (Mumbai), `ap-south-1`.
- **A strong API key for the server:**

  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```

## 1. Launch the EC2 instance (AWS console → EC2 → Launch instance)

| Setting | Value |
|---|---|
| Name | `sentinel-ai` |
| AMI | **Ubuntu Server 24.04 LTS** (x86_64) |
| Instance type | **m7i-flex.large** (2 vCPU, 8 GB). Pick one marked *Free tier eligible*. Smaller instances run out of memory on monitoring uploads |
| Key pair | Create new → `sentinel-ai` → `.pem`, then download and keep it safe |
| Network | Allow SSH from **My IP** only. Allow HTTPS and HTTP from the internet |
| Storage | **30 GiB gp3** |

Then go to **Elastic IPs → Allocate → Associate** with the instance. This gives a fixed public IP, so the HTTPS address doesn't change when you stop and start the instance.

## 2. Connect

On Windows PowerShell, first restrict the key file once, or SSH refuses it:

```powershell
icacls sentinel-ai.pem /inheritance:r /grant:r "$($env:USERNAME):R"
```

Then connect:

```powershell
ssh -i sentinel-ai.pem ubuntu@<ELASTIC_IP>
```

## 3. Install Docker and get the code (on the server)

```bash
git clone https://github.com/mohvijayjain/Sentinel-AI.git
cd Sentinel-AI
bash deploy/ec2_setup.sh
exit
```

SSH in again afterwards, so `docker` works without `sudo`.

## 4. Configure (on the server)

```bash
cd Sentinel-AI
cp deploy/.env.aws.example .env
nano .env
```

Fill in:
- **SITE_ADDRESS:** the Elastic IP with dashes, plus `.sslip.io`. For example, IP `13.201.5.20` becomes `13-201-5-20.sslip.io`.
- **POSTGRES_PASSWORD:** a new strong password, used both on its own line and inside `POSTGRES_URL`.
- **API_KEY:** the strong key from step 0.
- **NVIDIA_API_KEY:** needed for the AI Assistant.

## 5. Bring the laptop's data (monitoring history, retraining events, …)

**On the laptop** (PowerShell, repository root, with the local Docker stack running):

```powershell
.\deploy\export_local_data.ps1
scp -i sentinel-ai.pem -r sentinel-transfer ubuntu@<ELASTIC_IP>:~/
```

**On the server:**

```bash
cd ~/Sentinel-AI
bash deploy/import_data.sh
```

This extracts the data files, builds the images, restores PostgreSQL (and prints the row counts), restores MLflow artifacts and starts everything. The first build takes several minutes.

To start with an empty history instead, skip this step and run:

```bash
docker compose -f docker-compose.yml -f deploy/docker-compose.aws.yml up -d --build
```

Even then, the reference data has to be on the server for monitoring uploads: copy `data.tgz` and extract it first.

## 6. Check

```bash
curl https://<SITE_ADDRESS>/health
```

Expected response: `{"status":"healthy","model_loaded":true}`. HTTPS can take a minute the first time, while Caddy gets the certificate.

## 7. Connect the Streamlit Cloud UI

In your Streamlit Cloud app, go to **Manage app → ⋮ → Settings → Secrets** and add:

```toml
SENTINEL_API_URL = "https://<SITE_ADDRESS>"
SENTINEL_API_KEY = "<the API_KEY from the server's .env>"
```

Save and reboot the app. The banners disappear and the sidebar shows **Backend: ● Connected**.

## Everyday operations (on the server, in `~/Sentinel-AI`)

Set an alias for the long compose command first:

```bash
alias dc='docker compose -f docker-compose.yml -f deploy/docker-compose.aws.yml'
```

| Task | Command |
|---|---|
| Status | `dc ps` |
| API logs | `dc logs -f sentinel-api` |
| Deploy new code | `git pull && dc up -d --build sentinel-api` |
| Apply `.env` changes | `dc up -d --force-recreate sentinel-api` |
| Restart everything | `dc up -d` |

Never run `dc down -v`: it deletes the database volume.

## Cost

- **m7i-flex.large:** about $70/month if it runs 24×7, paid from the Free-plan credits.
- **Elastic IP and 30 GB disk:** a few dollars a month.
- **Saving credits:** **Stop** the instance when you don't need it. The disk and Elastic IP keep the data and address.
- **Watch spending:** check the Billing dashboard, and keep the zero-spend budget alert on.
