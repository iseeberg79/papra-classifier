#!/usr/bin/env python3
"""
Retroactively classify existing Papra documents with Claude and apply tags via API.

Usage:
    python3 batch_classify_extended.py           # dry-run, shows what would be tagged
    python3 batch_classify_extended.py --apply   # actually create and apply tags
    python3 batch_classify_extended.py --skip-tagged  # skip docs that already have tags
"""

import argparse
import json
import os
import pathlib
import re
import sys
import threading
import anthropic
import requests


def _load_env(path: str | None = None):
    env_file = pathlib.Path(path) if path else pathlib.Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)

PAPRA_BASE  = os.environ.get("PAPRA_BASE",  "https://app.papra.app")
PAPRA_TOKEN = os.environ.get("PAPRA_TOKEN", "")
ORG_ID      = os.environ.get("PAPRA_ORG_ID", "")

_SYSTEM_PROMPT = """\
Du klassifizierst Dokumente und extrahierst strukturierte Metadaten.
Antworte ausschließlich mit JSON in diesem Format:

{
  "ordner": "Ordnerkategorie",
  "typ": "Dokumenttyp",
  "schlagworte": ["keyword1", "keyword2"],
  "datum": "YYYY-MM oder leer",
  "jahr": "YYYY oder leer"
}

Regeln:

ordner — wähle aus dieser Tabelle; nur wenn kein Ordner passt und der neue für viele
  zukünftige Dokumente taugt, darf ein neuer Ordnername vorgeschlagen werden:
  Finanzen:    Rechnung, Angebot, Mahnung, Kontoauszug, Steuerbescheid, Lohnabrechnung, Quittung
  Verträge:    Vertrag, Kündigung, Versicherungsschein, Mietvertrag, Kaufvertrag
  Logistik:    Bestellung, Bestellbestätigung, Lieferschein, Sendungsverfolgung
  Technik:     Planung, Auslegung, Datenblatt, Handbuch, Protokoll, Zertifikat, Gutachten
  Energie:     Netzanmeldung, Einspeisevertrag, Inbetriebnahme, Messkonzept
  Sonstiges:   Bescheid, Antrag, Korrespondenz, Bericht

typ — wähle den passendsten Dokumenttyp aus der Spalte des gewählten Ordners.

schlagworte — 0–2 stabile Oberbegriffe für die Aktenablage.
  Bevorzuge vorhandene Tags. Wähle Oberbegriffe (z.B. "Kraftstoff" statt "Diesel").
  Nur Begriffe, unter denen viele Dokumente dauerhaft abgelegt werden können.

datum — nur Jahr-Monat (YYYY-MM, kein Tag), z.B. "2024-05", sonst leer
jahr  — Urkundenjahr vierstellig wenn erkennbar, sonst leer\
"""

_COLORS = [
    "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
    "#ec4899", "#06b6d4", "#f97316", "#84cc16", "#6366f1",
]

_tag_cache: dict[str, str] = {}
_tag_names: dict[str, str] = {}
_tag_lock = threading.Lock()
_anthropic_client: anthropic.Anthropic | None = None


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


def _papra(method: str, path: str, **kwargs):
    return requests.request(
        method,
        f"{PAPRA_BASE}/api/organizations/{ORG_ID}{path}",
        headers={"Authorization": f"Bearer {PAPRA_TOKEN}"},
        timeout=15,
        **kwargs,
    )


def _list_all_documents() -> list[dict]:
    r = _papra("GET", "/documents")
    r.raise_for_status()
    body = r.json()
    return body.get("documents") or body.get("data") or []


def _fetch_doc_content(doc_id: str) -> tuple[str, str]:
    """Returns (content, createdAt)."""
    try:
        r = _papra("GET", f"/documents/{doc_id}")
        r.raise_for_status()
        body = r.json()
        doc = body.get("document") or body.get("data") or {}
        return doc.get("content") or "", doc.get("createdAt") or ""
    except Exception as e:
        print(f"  [warn] fetch content {doc_id}: {e}", flush=True)
        return "", ""


def _claude_tags(name: str, content: str) -> tuple[list[str], str]:
    """Returns (tags, datum) — datum is YYYY-MM for documentDate, not a tag."""
    try:
        system = [{"type": "text", "text": _SYSTEM_PROMPT}]
        if _tag_names:
            system.append({
                "type": "text",
                "text": f"Vorhandene Tags (bevorzuge passende):\n{', '.join(sorted(_tag_names.values()))}",
                "cache_control": {"type": "ephemeral"},
            })
        else:
            system[0]["cache_control"] = {"type": "ephemeral"}

        response = _get_anthropic().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": f"Name: {name}\n\nInhalt:\n{content[:3000]}"}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        text = re.sub(r"^```[a-z]*\n?", "", text.strip()).rstrip("`").strip()
        r = json.loads(text)
        datum = re.sub(r'^(\d{4}-\d{2})-\d{2}$', r'\1', r.get("datum") or "")
        tags = []
        if r.get("ordner"):
            tags.append(r["ordner"])
        if r.get("typ") and r.get("typ") != r.get("ordner"):
            tags.append(r["typ"])
        tags.extend(r.get("schlagworte") or [])
        if r.get("jahr") and r.get("jahr") not in tags:
            tags.append(r["jahr"])
        return tags, datum
    except Exception as e:
        print(f"  [warn] claude classify: {e}", flush=True)
        return [], ""


def _set_doc_date(doc_id: str, datum: str, created_at: str, apply: bool):
    if not apply:
        return
    if datum:
        date_str = datum + "-01"
    elif created_at:
        date_str = created_at[:10]
    else:
        return
    try:
        r = _papra("PATCH", f"/documents/{doc_id}", json={"documentDate": date_str})
        if not r.ok:
            print(f"  [warn] set date {doc_id}: {r.status_code}", flush=True)
    except Exception as e:
        print(f"  [warn] set date {doc_id}: {e}", flush=True)


def _populate_tag_cache():
    try:
        r = _papra("GET", "/tags")
        r.raise_for_status()
        body = r.json()
        tags = body.get("tags") or body.get("data") or []
        with _tag_lock:
            for t in tags:
                _tag_cache[t["name"].lower()] = t["id"]
                _tag_names[t["name"].lower()] = t["name"]
    except Exception as e:
        print(f"[warn] load tags: {e}", flush=True)


def _get_or_create_tag(label: str, apply: bool) -> str | None:
    key = label.lower()
    with _tag_lock:
        if key in _tag_cache:
            return _tag_cache[key]

    if not apply:
        return f"(dry-run:{label})"

    color = _COLORS[sum(ord(c) for c in label) % len(_COLORS)]
    try:
        r = _papra("POST", "/tags", json={"name": label, "color": color})
        if not r.ok:
            print(f"  [warn] create tag '{label}': {r.status_code} {r.text}", flush=True)
            return None
        body = r.json()
        tag = body.get("tag") or body.get("data") or {}
        tag_id = tag.get("id")
        if tag_id:
            with _tag_lock:
                _tag_cache[key] = tag_id
                _tag_names[key] = label
        return tag_id
    except Exception as e:
        print(f"  [warn] create tag '{label}': {e}", flush=True)
        return None


def _apply_tag(doc_id: str, tag_id: str, label: str, apply: bool):
    if not apply:
        return
    try:
        r = _papra("POST", f"/documents/{doc_id}/tags", json={"tagId": tag_id})
        if r.status_code == 409:
            return  # already set
        r.raise_for_status()
    except Exception as e:
        print(f"  [warn] apply tag '{label}': {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Batch-classify Papra documents with Claude")
    parser.add_argument("--apply", action="store_true", help="Actually create and apply tags (default: dry-run)")
    parser.add_argument("--skip-tagged", action="store_true", help="Skip documents that already have tags")
    parser.add_argument("--env-file", metavar="PATH", help="Path to .env file (default: .env next to this script)")
    args = parser.parse_args()

    _load_env(args.env_file)

    missing = [v for v in ("ANTHROPIC_API_KEY", "PAPRA_TOKEN", "PAPRA_ORG_ID") if not os.environ.get(v)]
    if missing:
        print(f"Error: missing env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== batch_classify_extended [{mode}] ===", flush=True)

    print("Loading existing tags...", flush=True)
    _populate_tag_cache()
    print(f"  {len(_tag_cache)} tags cached", flush=True)

    print("Listing documents...", flush=True)
    try:
        docs = _list_all_documents()
    except Exception as e:
        print(f"Error listing documents: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(docs)} documents found", flush=True)

    for i, doc in enumerate(docs, 1):
        doc_id = doc.get("id") or ""
        name   = doc.get("name") or doc.get("originalName") or ""
        existing_tags = doc.get("tags") or []

        if not doc_id or not name:
            continue

        if args.skip_tagged and existing_tags:
            print(f"[{i}/{len(docs)}] SKIP {name!r} ({len(existing_tags)} tags)", flush=True)
            continue

        print(f"[{i}/{len(docs)}] {name!r}", flush=True)
        content, created_at = _fetch_doc_content(doc_id)
        tags, datum = _claude_tags(name, content)

        if not tags:
            print(f"  → no tags from Claude", flush=True)
            continue

        date_src = datum or created_at[:10]
        print(f"  → {tags}  date={date_src or '?'}", flush=True)
        for label in tags:
            tag_id = _get_or_create_tag(label, args.apply)
            if tag_id:
                _apply_tag(doc_id, tag_id, label, args.apply)
        _set_doc_date(doc_id, datum, created_at, args.apply)

    print(f"\nDone. {len(docs)} documents processed.", flush=True)
    if not args.apply:
        print("Re-run with --apply to actually write tags.", flush=True)


if __name__ == "__main__":
    main()
