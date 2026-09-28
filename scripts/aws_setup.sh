#!/usr/bin/env bash
# DocuMind RAG — AWS EC2 One-Touch Setup & Deployment Script
set -e

echo "===================================================================="
echo "🚀 Starting DocuMind RAG Setup on AWS EC2 Ubuntu Server..."
echo "===================================================================="

# 1. Update OS packages & ensure swap memory for builds
sudo apt-get update -y
sudo apt-get install -y curl git apt-transport-https ca-certificates software-properties-common

if [ $(free -m | awk '/^Mem:/{print $2}') -lt 3000 ] && [ ! -f /swapfile ]; then
    echo "🧠 Allocating 2GB swap space for low-memory EC2 instance..."
    sudo fallocate -l 2G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
fi

# 2. Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker Engine..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

# 3. Install Docker Compose plugin & Caddy
echo "📦 Installing Docker Compose & Caddy Reverse Proxy..."
sudo apt-get install -y docker-compose-plugin caddy

# 4. Verify Docker daemon running
sudo systemctl enable docker
sudo systemctl start docker

# 5. Build and launch Docker Compose stack
echo "🐳 Launching DocuMind Docker containers (DB + API + Streamlit)..."
docker compose up -d --build

# 6. Wait for Database readiness
echo "⏳ Waiting for pgvector database to complete health check..."
until docker compose exec -T db pg_isready -U documind -d documind; do
  echo "Waiting for database..."
  sleep 2
done

# 7. Seed Knowledge Base Corpus
echo "📚 Ingesting initial enterprise document corpus..."
docker compose exec -T api python scripts/ingest_documents.py --source ./data/sample_docs

# 8. Start Caddy Reverse Proxy
echo "🔒 Starting Caddy reverse proxy..."
sudo caddy stop 2>/dev/null || true
sudo caddy start --config ./Caddyfile

PUBLIC_IP=$(curl -s ifconfig.me || echo "your-ec2-public-ip")

echo "===================================================================="
echo "🎉 DocuMind RAG Deployment Complete!"
echo "===================================================================="
echo "🌐 Streamlit UI:        http://${PUBLIC_IP}"
echo "🔌 API Docs (Swagger):  http://${PUBLIC_IP}:8000/docs"
echo "📊 Health Metric:       http://${PUBLIC_IP}:8000/health"
echo "===================================================================="
