# Deploying Brain to a DigitalOcean Droplet

This guide walks through deploying the Brain FastAPI service to a fresh
DigitalOcean droplet, fronted by Nginx with a Let's Encrypt TLS cert and
managed by `systemd`. The service runs as `root` from `/root/Brain` — fine
for a single-purpose droplet.

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

Once it boots, note the public IPv4. In your DNS provider (this project
uses **Cloudflare**), add an A record pointing `brain.jayprajapati.dev` at
the droplet IP:

- **Type:** A
- **Name:** `brain`
- **IPv4:** `<droplet public IP>`
- **Proxy status:** **DNS only** (gray cloud) — required for the initial
  Let's Encrypt HTTP-01 challenge. You can flip it to proxied later if you
  also switch Cloudflare SSL mode to **Full (strict)**.

Verify before continuing:

```bash
dig +short brain.jayprajapati.dev
```

Should print the droplet IP. If you get nothing or a `172.x.x.x` (Cloudflare
edge), wait a minute or fix the proxy status.

---

## 2. Initial server hardening

SSH in as root and set up the firewall:

```bash
ssh root@<droplet-ip>

ufw allow OpenSSH
ufw allow 80
ufw allow 443
ufw enable

# Lock SSH to key auth only
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart ssh
```

---

## 3. Install system dependencies

```bash
apt update && apt upgrade -y
apt install -y python3.12 python3.12-venv python3-pip \
               git nginx certbot python3-certbot-nginx
```

If the droplet has < 4 GB RAM, add a 2 GB swapfile so the first ONNX
download doesn't get killed:

```bash
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

---

## 4. Clone the repo and set up the venv

```bash
cd /root
git clone <your-repo-url> Brain
cd Brain

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
python -c "from app.embeddings import warmup; warmup()"
python -c "from app.reranker import warmup; warmup()"
```

This caches the ONNX files under `/root/.cache/fastembed/`.

---

## 7. Create the systemd service

Open the unit file in nano and paste:

```bash
nano /etc/systemd/system/brain.service
```

Paste this content (`Ctrl+O` → `Enter` to save, `Ctrl+X` to exit):

```ini
[Unit]
Description=Brain RAG API
After=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/root/Brain
EnvironmentFile=/root/Brain/.env
ExecStart=/root/Brain/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8010 --workers 1 --proxy-headers --forwarded-allow-ips=*
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

After saving, run `cat /etc/systemd/system/brain.service` to confirm every
line starts flush-left with no leading spaces (some terminals add indent on
paste — if that happened, re-open and fix).

Worker count rule of thumb: **1 worker per GB of RAM**, capped at vCPU count.
Each worker loads its own embedding model. On a 2 GB / 1 vCPU droplet keep it
at `--workers 1`; bump up on a larger droplet.

Enable and start:

```bash
systemctl daemon-reload
systemctl enable --now brain
systemctl status brain --no-pager
journalctl -u brain -f   # tail logs (Ctrl-C to exit)
```

Sanity check it's listening locally:

```bash
curl http://127.0.0.1:8010/health
```

---

## 8. Nginx reverse proxy

Open the config in nano:

```bash
nano /etc/nginx/sites-available/brain
```

Paste this (`Ctrl+O` → `Enter`, `Ctrl+X`):

```nginx
server {
    listen 80;
    server_name brain.jayprajapati.dev;

    client_max_body_size 25M;

    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 600s;

    location / {
        proxy_pass http://127.0.0.1:8010;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection '';
    }
}
```

Then enable it:

```bash
ln -sf /etc/nginx/sites-available/brain /etc/nginx/sites-enabled/brain
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx
```

The SSE bits (`proxy_buffering off`, `Connection ''`, long timeout) are
essential — without them `/v1/chat` will hang or close early.

---

## 9. TLS with Let's Encrypt

Make sure Cloudflare proxy is **off** (gray cloud) for the `brain` record
before running this — otherwise the HTTP-01 challenge hits Cloudflare's
edge instead of your droplet and fails with an auth error.

```bash
certbot --nginx -d brain.jayprajapati.dev
```

Certbot rewrites the Nginx config to listen on 443 and sets up a renewal
timer. Verify:

```bash
curl https://brain.jayprajapati.dev/health
systemctl list-timers | grep certbot
```

After the cert is issued you can re-enable the Cloudflare proxy (orange
cloud) — but **only** if you also set Cloudflare → SSL/TLS → Overview to
**Full (strict)**. Anything else either breaks the connection or strips
end-to-end encryption.

---

## 10. Smoke test from your machine

```bash
curl https://brain.jayprajapati.dev/health

curl -X POST https://brain.jayprajapati.dev/v1/llm/ping \
  -H "Authorization: Bearer $BRAIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"llm":{"provider":"ollama_cloud"}}'
```

Then point the portfolio backend at it by setting `BRAIN_BASE_URL=https://brain.jayprajapati.dev`
and `BRAIN_API_KEY=<the secret you generated>` in its env.

---

## 11. Deploying updates

```bash
ssh root@<droplet-ip>
cd /root/Brain
git pull
source .venv/bin/activate
pip install -r requirements.txt   # only if requirements.txt changed
systemctl restart brain
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
