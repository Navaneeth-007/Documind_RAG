# DocuMind — Deployment & Hosting Guide

This guide covers options for deploying DocuMind into production or staging environments.

---

## Architecture Overview

```
                      ┌─────────────────────────┐
                      │    Streamlit Frontend   │ (Port 8501)
                      └────────────┬────────────┘
                                   │ HTTP (API_URL)
                      ┌────────────▼────────────┐
                      │    FastAPI API Server   │ (Port 8000)
                      └────────────┬────────────┘
                                   │
                      ┌────────────▼────────────┐
                      │ PostgreSQL + pgvector   │ (Port 5432)
                      └─────────────────────────┘
```

---

## Option 1: AWS EC2 Instance (Recommended for Portfolio & Production Showcase)

Deploying on **AWS EC2** with **Docker Compose** and **Caddy Reverse Proxy** demonstrates full-stack DevOps engineering, cloud security, and container orchestration skills.

### Step 1: Launch an AWS EC2 Instance
1. Go to **AWS Management Console** → **EC2** → **Launch Instance**.
2. **Name**: `documind-rag-server`
3. **AMI**: `Ubuntu Server 24.04 LTS` (64-bit x86)
4. **Instance Type**: `t3.small` (2 vCPU, 2GB RAM) or `t2.micro` (Free Tier)
5. **Key Pair**: Select or create a `.pem` SSH key.
6. **Network / Security Group**:
   - Allow **SSH** (`22`) from your IP.
   - Allow **HTTP** (`80`) from Anywhere (`0.0.0.0/0`).
   - Allow **HTTPS** (`443`) from Anywhere (`0.0.0.0/0`).
   - Allow **Custom TCP** (`8000`) from Anywhere (`0.0.0.0/0`).
7. Click **Launch Instance**.

---

### Step 2: SSH into EC2 & Clone Repository
```bash
ssh -i /path/to/your-key.pem ubuntu@<EC2-PUBLIC-IP>

# Clone repository
git clone https://github.com/your-username/Documind-RAG.git
cd Documind-RAG
```

---

### Step 3: Configure Environment Variables
Create your production `.env` file on EC2:
```bash
nano .env
```
Paste your production settings:
```env
POSTGRES_USER=documind
POSTGRES_PASSWORD=documind_secure_pass_2026
POSTGRES_DB=documind
DATABASE_URL=postgresql://documind:documind_secure_pass_2026@db:5432/documind

LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=openai/gpt-oss-120b

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

### Step 4: Run Automated AWS One-Touch Setup Script
Execute the provided automated deployment script:
```bash
chmod +x scripts/aws_setup.sh
./scripts/aws_setup.sh
```

This script will automatically:
1. Install Docker Engine & Docker Compose Plugin.
2. Build and launch PostgreSQL + `pgvector`, FastAPI, and Streamlit containers.
3. Wait for database health checks and initialize vector tables with HNSW indexing.
4. Ingest sample documents into the knowledge base.
5. Start Caddy reverse proxy on port 80/443.

---

### Step 5: Access your AWS Deployment
- 🌐 **Streamlit UI**: `http://<EC2-PUBLIC-IP>`
- 🔌 **API Documentation**: `http://<EC2-PUBLIC-IP>:8000/docs`
- 📊 **Health Metrics**: `http://<EC2-PUBLIC-IP>:8000/health`

---

## Option 2: Render.com (Easiest 1-Click Blueprint)

DocuMind includes a `render.yaml` infrastructure-as-code specification.

1. Push your repository to GitHub.
2. Log into [Render](https://render.com).
3. Click **Blueprints** → **New Blueprint Instance**.
4. Connect your GitHub repository. Render will detect `render.yaml` and provision:
   - Managed PostgreSQL database (`documind-db`) with vector support.
   - FastAPI backend (`documind-api`).
   - Streamlit frontend (`documind-frontend`).
5. Enter your `OPENAI_API_KEY` (or Anthropic / Groq key) in the dashboard secrets.
6. Click **Apply**. Render will build and deploy the entire stack with free HTTPS endpoints.

---

## Option 2: Railway.app

1. Fork or push this repository to GitHub.
2. In [Railway](https://railway.app), click **New Project** → **Deploy from GitHub repo**.
3. Add a **PostgreSQL** service from the Railway template gallery.
4. Enable the `pgvector` extension in the Postgres query console:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
5. Deploy the **API Service**:
   - Set Build Command / Dockerfile target.
   - Set Environment Variables:
     - `DATABASE_URL`: `${{Postgres.DATABASE_URL}}`
     - `OPENAI_API_KEY`: `your-api-key`
     - `LLM_PROVIDER`: `openai`
     - `PORT`: `8000`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
6. Deploy the **Frontend Service**:
   - Environment Variables:
     - `API_URL`: `https://${{api.RAILWAY_PUBLIC_DOMAIN}}`
   - Start Command: `streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0`

---

## Option 3: Single VPS / Cloud Instance (Docker Compose + Caddy SSL)

For deploying on AWS EC2, DigitalOcean Droplet, Hetzner, or Linode ($5–$10/mo):

### 1. Clone & Configure
```bash
git clone https://github.com/your-username/documind-rag.git
cd documind-rag
nano .env  # fill in OPENAI_API_KEY or other provider credentials
```

### 2. Ingest Initial Knowledge Base
```bash
docker compose up -d db
docker compose run --rm api python scripts/ingest_documents.py --source ./data/sample_docs
```

### 3. Launch Full Production Stack
```bash
docker compose up -d --build
```

### 4. Automatic HTTPS Reverse Proxy (Caddyfile)
Create `Caddyfile`:
```caddy
your-domain.com {
    reverse_proxy localhost:8501
}

api.your-domain.com {
    reverse_proxy localhost:8000
}
```
Run `caddy run` to get automatic Let's Encrypt SSL certificates.

---

## Option 4: Hugging Face Spaces (Frontend Only)

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) selecting **Streamlit**.
2. Upload the `frontend/` directory and `requirements.txt`.
3. In Space **Settings** → **Variables and Secrets**, add:
   - `API_URL`: URL of your hosted FastAPI backend.
4. Hugging Face will automatically host your interactive chat UI.

---

## Health Checks & Diagnostics

After deployment, verify that your API and database are functioning:
- **API Health:** `GET https://<your-api-url>/health`
- **Interactive Swagger Docs:** `GET https://<your-api-url>/docs`
- **Corpus List:** `GET https://<your-api-url>/documents`
- **LLMOps Telemetry:** `GET https://<your-api-url>/analytics`
