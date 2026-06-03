# Deploying Brain to a DigitalOcean Droplet

This guide walks through deploying the Brain FastAPI service to a fresh
DigitalOcean droplet, fronted by Nginx with a Let's Encrypt TLS cert and
managed by `systemd`.

Brain is a thin stateless API — Qdrant Cloud and Ollama Cloud are external,
so the droplet only runs the Python process. The first start downloads the
embedding + reranker ONNX models (~500 MB), so size accordingly.

---

## 1. Provision the droplet

In the DigitalOcean dashboard:

- **Image:** Ubuntu 24.04 LTS
- **Plan:** Basic, regular SSD. Minimum **2 GB RAM / 1 vCPU** (`s-1vcpu-2gb`).
  fastembed loads ONNX models into memory; 1 GB will OOM during first inference.
- **Region:** closest to your Qdrant Cloud region (cuts retrieval latency).
- **Auth:** SSH key (paste your public key).
- **Hostname:** e.g. `brain-prod`.

Once it boots, note the public IPv4. Point an A record at it
(e.g. `brain.yourdomain.com`) — you'll need this for TLS later.

---

## 2. Initial server hardening

SSH in as root, then create a non-root user and lock things down:

```bash
ssh root@<droplet-ip>

adduser brain
usermod -aG sudo brain
rsync --archive --chown=brain:brain ~/.ssh /home/brain

ufw allow OpenSSH
ufw allow 80
ufw allow 443
ufw enable

# Disable root SSH and password auth
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart ssh
```

From here on, log in as `brain`: `ssh brain@<droplet-ip>`.

---

## 3. Install system dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.12 python3.12-venv python3-pip \
                    git nginx certbot python3-certbot-nginx
```

If the droplet has < 4 GB RAM, add a 2 GB swapfile so the first ONNX
download doesn't get killed:

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## 4. Clone the repo and set up the venv

```bash
sudo mkdir -p /opt/brain
sudo chown brain:brain /opt/brain
cd /opt/brain

git clone <your-repo-url> .
# or for a private repo, deploy key / PAT

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Configure `.env`

```bash
cp .env.example .env
nano .env
```

Fill in:

- `BRAIN_API_KEY` — generate a strong secret (`openssl rand -hex 32`).
  This is the bearer token every caller (e.g. the portfolio backend) must send.
- `OLLAMA_API_KEY` — from ollama.com (Brain's default LLM provider).
- `QDRANT_URL`, `QDRANT_API_KEY` — from Qdrant Cloud.
- `QDRANT_COLLECTION_PREFIX` — optional namespace, e.g. `brain_` so portfolio
  vectors live in `brain_portfolio` not `portfolio`.

Lock it down:

```bash
chmod 600 .env
```

Verify the Ollama Cloud model is reachable:

```bash
python scripts/check_ollama.py
```

---

## 6. Pre-download the embedding models

Do this once interactively so the first real request isn't a 60-second
cold start (and so `systemd` doesn't think the service is hanging):

```bash
python -c "from app.embeddings import embed_texts; embed_texts(['warmup'])"
python -c "from app.reranker import rerank; rerank('q', [{'text':'a'}])"
```

This caches the ONNX files under `~/.cache/fastembed/`.

---

## 7. Create the systemd service

```bash
sudo nano /etc/systemd/system/brain.service
```

```ini
[Unit]
Description=Brain RAG API
After=network.target

[Service]
Type=simple
User=brain
Group=brain
WorkingDirectory=/opt/brain
EnvironmentFile=/opt/brain/.env
ExecStart=/opt/brain/.venv/bin/uvicorn app.main:app \
          --host 127.0.0.1 --port 8000 \
          --workers 2 --proxy-headers --forwarded-allow-ips='*'
Restart=on-failure
RestartSec=5
# fastembed + Qdrant client buffers eat memory; cap if you want to be safe
# MemoryMax=1500M

[Install]
WantedBy=multi-user.target
```

Worker count rule of thumb: **1 worker per GB of RAM**, capped at vCPU count.
Each worker loads its own embedding model.

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now brain
sudo systemctl status brain
journalctl -u brain -f   # tail logs
```

Sanity check it's listening locally:

```bash
curl http://127.0.0.1:8000/health
```

---

## 8. Nginx reverse proxy

```bash
sudo nano /etc/nginx/sites-available/brain
```

```nginx
server {
    listen 80;
    server_name brain.yourdomain.com;

    # Allow large multipart uploads for /v1/extract
    client_max_body_size 25M;

    # SSE needs buffering off and a long read timeout
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 600s;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection '';
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/brain /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

The SSE bits (`proxy_buffering off`, `Connection ''`, long timeout) are
essential — without them `/v1/chat` will hang or close early.

---

## 9. TLS with Let's Encrypt

```bash
sudo certbot --nginx -d brain.yourdomain.com
```

Certbot rewrites the Nginx config to listen on 443 and sets up a renewal
timer. Verify:

```bash
curl https://brain.yourdomain.com/health
sudo systemctl list-timers | grep certbot
```

---

## 10. Smoke test from your machine

```bash
curl https://brain.yourdomain.com/health

curl -X POST https://brain.yourdomain.com/v1/llm/ping \
  -H "Authorization: Bearer $BRAIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"llm":{"provider":"ollama_cloud"}}'
```

Then point the portfolio backend at it by setting `BRAIN_BASE_URL=https://brain.yourdomain.com`
and `BRAIN_API_KEY=<the secret you generated>` in its env.

---

## 11. Deploying updates

```bash
ssh brain@<droplet-ip>
cd /opt/brain
git pull
source .venv/bin/activate
pip install -r requirements.txt   # only if requirements.txt changed
sudo systemctl restart brain
journalctl -u brain -n 50 --no-pager
```

For zero-downtime updates, use `systemctl reload` only if you switch to
gunicorn with a worker class that supports SIGHUP; uvicorn's plain reload
isn't graceful. For Brain's traffic level, a 1-second restart is fine.

---

## Operational notes

- **Logs:** `journalctl -u brain -f`. Pipe to a file or hook into DO's
  monitoring agent if you want retention.
- **Metrics:** `/health` returns the active chat + embed models and provider
  status — point uptime monitoring at it.
- **Backups:** Brain itself is stateless on disk (vectors live in Qdrant
  Cloud). The only things worth backing up are `.env` and `/etc/systemd/system/brain.service`.
- **Scaling:** if you outgrow one droplet, put two behind a DO load balancer
  — Brain is stateless across requests, so no session affinity needed.
  Watch RAM before CPU; fastembed is memory-bound, not compute-bound.
- **Costs to watch:** Qdrant Cloud storage + Ollama Cloud tokens are the
  variable bills. The droplet is fixed.
