import fasttext
import json
import os
import re
import threading
import anthropic
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)
model = fasttext.load_model("papra_model.bin")

PAPRA_BASE       = os.environ.get("PAPRA_BASE",  "https://app.papra.app")
PAPRA_TOKEN      = os.environ["PAPRA_TOKEN"]
ORG_ID           = os.environ["PAPRA_ORG_ID"]
USE_CLAUDE       = os.environ.get("CLASSIFIER_MODE", "claude").lower() == "claude"

_tag_cache: dict[str, str] = {}   # label.lower() -> tagId
_tag_names: dict[str, str] = {}   # label.lower() -> original label
_tag_lock = threading.Lock()

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

_anthropic_client: anthropic.Anthropic | None = None


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


def _fetch_doc_content(doc_id: str) -> tuple[str, str]:
    """Returns (content, createdAt)."""
    try:
        r = _papra("GET", f"/documents/{doc_id}")
        r.raise_for_status()
        body = r.json()
        doc = body.get("document") or body.get("data") or {}
        return doc.get("content") or "", doc.get("createdAt") or ""
    except Exception as e:
        print(f"[papra] fetch doc {doc_id}: {e}", flush=True)
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
        print(f"[claude] classify error: {e}", flush=True)
        return [], ""


def _set_doc_date(doc_id: str, datum: str, created_at: str):
    if datum:
        date_str = datum + "-01"          # YYYY-MM → YYYY-MM-01
    elif created_at:
        date_str = created_at[:10]        # ISO datetime → YYYY-MM-DD
    else:
        return
    try:
        r = _papra("PATCH", f"/documents/{doc_id}", json={"documentDate": date_str})
        if not r.ok:
            print(f"[papra] set date {doc_id}: {r.status_code}", flush=True)
    except Exception as e:
        print(f"[papra] set date {doc_id}: {e}", flush=True)


def _papra(method: str, path: str, **kwargs):
    return requests.request(
        method,
        f"{PAPRA_BASE}/api/organizations/{ORG_ID}{path}",
        headers={"Authorization": f"Bearer {PAPRA_TOKEN}"},
        timeout=30,
        **kwargs,
    )


def _get_or_create_tag(label: str) -> str | None:
    key = label.lower()
    with _tag_lock:
        if key in _tag_cache:
            return _tag_cache[key]

    try:
        r = _papra("GET", "/tags")
        r.raise_for_status()
        body = r.json()
        tags = body.get("tags") or body.get("data") or []
        with _tag_lock:
            for t in tags:
                _tag_cache[t["name"].lower()] = t["id"]
                _tag_names[t["name"].lower()] = t["name"]
            if key in _tag_cache:
                return _tag_cache[key]
    except Exception as e:
        print(f"[papra] get tags: {e}", flush=True)
        return None

    _COLORS = [
        "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
        "#ec4899", "#06b6d4", "#f97316", "#84cc16", "#6366f1",
    ]
    color = _COLORS[sum(ord(c) for c in label) % len(_COLORS)]

    try:
        r = _papra("POST", "/tags", json={"name": label, "color": color})
        if not r.ok:
            print(f"[papra] create tag '{label}': {r.status_code} {r.text}", flush=True)
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
        print(f"[papra] create tag '{label}': {e}", flush=True)
        return None


def _apply_tag(doc_id: str, tag_id: str, label: str):
    try:
        r = _papra("POST", f"/documents/{doc_id}/tags", json={"tagId": tag_id})
        if r.status_code == 409:
            print(f"[papra] {doc_id} → tag '{label}' bereits gesetzt", flush=True)
            return
        r.raise_for_status()
        print(f"[papra] {doc_id} → tag '{label}' gesetzt", flush=True)
    except Exception as e:
        print(f"[papra] apply tag {doc_id}: {e}", flush=True)


def _classify_and_tag(doc_id: str, name: str):
    content, created_at = _fetch_doc_content(doc_id)
    datum = ""
    if USE_CLAUDE:
        tags, datum = _claude_tags(name, content)
    else:
        tags = []
    if not tags:
        text = name + " " + content[:2000] if content else name
        tags = classify_text(text)
    print(f"[classify] {doc_id} | {name!r} → {tags}", flush=True)
    for label in tags:
        tag_id = _get_or_create_tag(label)
        if tag_id:
            _apply_tag(doc_id, tag_id, label)
    _set_doc_date(doc_id, datum, created_at)


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s,.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


RULES = [
    ("vertrag",      ["agb", "allgemeine geschäftsbedingungen", "nutzungsbedingungen",
                      "vertrag", "mietvertrag", "arbeitsvertrag", "kaufvertrag", "lizenzvertrag",
                      "vertragsbestätigung", "kündigung", "laufzeit"]),
    ("rechnung",     ["rechnung", "invoice", "faktura", "nettobetrag", "bruttobetrag",
                      "umsatzsteuer", "zahlungsziel", "rechnungsnummer", "abrechnung"]),
    ("bestellung",   ["bestellbestätigung", "bestellnummer", "sendungsverfolgung",
                      "trackingnummer", "lieferstatus", "auftragsbestätigung"]),
    ("versicherung", ["versicherungsschein", "policennummer", "versicherungsbestätigung",
                      "prämie", "selbstbeteiligung"]),
    ("bank",         ["kontoauszug", "buchungsdatum", "kontonummer", "depotauszug"]),
    ("Netzanmeldung",  ["netzanmeldung", "anschlussbegehren", "marktstammdaten"]),
    ("Einspeisevertrag", ["einspeisevertrag", "einspeisevergütung", "einspeisetarif"]),
    ("Inbetriebnahme", ["inbetriebnahmeprotokoll", "erstinbetriebnahme"]),
    ("Messkonzept",    ["messkonzept", "zweirichtungszähler", "standardlastprofil"]),
]

def rule_based_type(text):
    haystack = text.lower()
    for doc_type, keywords in RULES:
        if any(re.search(r'\b' + re.escape(kw) + r'\b', haystack) for kw in keywords):
            return doc_type
    return None


def classify_text(text) -> list[str]:
    clean = preprocess(text)
    labels, probs = model.predict(clean, k=10)
    result = [
        l.replace("__label__", "")
        for l, p in zip(labels, probs)
        if p >= 0.2
    ]
    if result:
        return result
    rule = rule_based_type(text)
    return [rule.capitalize()] if rule else [labels[0].replace("__label__", "")]



@app.post("/classify")
def classify():
    try:
        data = request.get_json(force=True)
        inner  = data.get("data") or data
        doc_id = inner.get("documentId") or ""
        name   = inner.get("name") or ""
        if not doc_id or not name:
            return jsonify({"error": "missing documentId or name"}), 400
        threading.Thread(target=_classify_and_tag, args=(doc_id, name), daemon=True).start()
        return jsonify({"ok": True}), 200
    except Exception as e:
        print(f"[classify] error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
