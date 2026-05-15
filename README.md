# papra-classifier

Automatic document classification and tagging for [Papra](https://github.com/papra-hq/papra) — triggered by webhook, powered by Claude or FastText.

**Use case:** Every document uploaded to Papra is automatically classified and tagged with document type, folder category, and thematic keywords — without manual intervention.

## How it works

```
Papra (new document)
  → POST /classify  (webhook)
  → fetch document content via Papra API
  → Claude (rich: type + folder + keywords + date)
     or FastText (local: type + folder)
  → create/apply tags via Papra API
  → set documentDate via Papra API
```

### Taxonomy

Documents are classified using a two-level DMS structure:

| Ordner (Folder) | Typ (Document Type)                                                    |
|-----------------|------------------------------------------------------------------------|
| Finanzen        | Rechnung, Angebot, Mahnung, Kontoauszug, Steuerbescheid, Lohnabrechnung, Quittung |
| Verträge        | Vertrag, Kündigung, Versicherungsschein, Mietvertrag, Kaufvertrag      |
| Logistik        | Bestellung, Bestellbestätigung, Lieferschein, Sendungsverfolgung       |
| Technik         | Planung, Auslegung, Datenblatt, Handbuch, Protokoll, Zertifikat, Gutachten |
| Energie         | Netzanmeldung, Einspeisevertrag, Inbetriebnahme, Messkonzept           |
| Sonstiges       | Bescheid, Antrag, Korrespondenz, Bericht                               |

### Classifier modes

| Mode       | Backend                        | Tags                              | Cost    |
|------------|--------------------------------|-----------------------------------|---------|
| `claude`   | Claude Haiku (Anthropic API)   | Folder + Type + Keywords + Date   | API fee |
| `fasttext` | Local FastText model           | Folder + Type                     | Free    |

In `claude` mode, FastText is used as fallback if Claude fails or returns no result.

## Requirements

- Docker + Docker Compose
- Papra instance with webhook support
- For `claude` mode: Anthropic API key
- For `fasttext` mode: trained model (see below)

## Setup

### 1. Create `.env`

```bash
cp .env.example .env
```

Edit `.env`:

```env
PAPRA_TOKEN=ppapi_...          # Papra API token (read/write documents and tags)
PAPRA_ORG_ID=org_...           # Papra organisation ID
ANTHROPIC_API_KEY=sk-ant-...   # Only required for CLASSIFIER_MODE=claude
CLASSIFIER_MODE=claude         # claude | fasttext
WEBHOOK_SECRET=...             # Must match the secret set in Papra (see step 3)
PAPRA_BASE=https://your-papra.example.com  # required for self-hosted; omit for app.papra.app
```

### 2. Train the FastText model (fasttext mode only)

```bash
# Debian/Ubuntu
sudo apt install python3-fasttext

# or via pip
pip install fasttext-wheel

python3 train_model.py
```

This reads `train.txt` and writes `papra_model.bin`. The included `train.txt` covers the taxonomy above with German document vocabulary — extend it with your own examples for better accuracy.

### 3. Configure Papra webhook

In the Papra web UI, navigate to your organisation → **Settings → Webhooks → Create webhook**:

| Field   | Value                                                                                      |
|---------|--------------------------------------------------------------------------------------------|
| Name    | Classifier (or any name)                                                                   |
| URL     | `https://<classifier-reverse-proxy-host.papra-domain.tld>/classify`                        |
| Secret  | Generate a random string and set the same value as `WEBHOOK_SECRET` in `.env`              |
| Events  | ✅ `document:created`                                                                      |

Generate a secret:
```bash
openssl rand -hex 32
```

### 4. Build and run

```bash
docker compose up -d --build
```

### Reverse proxy (optional)

To expose the classifier on the same domain as Papra (e.g. via Traefik), see the commented `labels` section in `docker-compose.yaml`. The classifier will then be reachable at `https://your-papra.example.com/classifier/classify` — update the Papra webhook URL accordingly.

## Batch classification

To retroactively classify existing documents:

```bash
python3 batch_classify.py --env-file .env           # dry-run
python3 batch_classify.py --env-file .env --apply   # apply tags
python3 batch_classify.py --env-file .env --apply --skip-tagged  # skip already tagged
```

## Files

| File                        | Purpose                                       |
|-----------------------------|-----------------------------------------------|
| `classify.py`               | Flask service — webhook endpoint + classifier |
| `batch_classify.py`         | Retroactive batch classification              |
| `train_model.py`            | FastText model training                       |
| `train.txt`                 | Training data (German DMS vocabulary)         |
| `Dockerfile`                | Container image                               |
| `docker-compose.yaml`       | Service definition                            |
