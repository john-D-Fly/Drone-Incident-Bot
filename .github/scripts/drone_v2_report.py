#!/usr/bin/env python3
"""
Drone News V2 — reads your incident JSON files, asks ChatGPT to cluster/score/summarize,
then writes:
  - reports/YYYY-MM-DD.md   (human-readable brief)
  - reports/YYYY-MM-DD.json (structured output)

ENV:
  - OPENAI_API_KEY   (required)
  - OPENAI_MODEL     (optional; default 'gpt-4o-mini')

CLI:
  python .github/scripts/drone_v2_report.py --data-dir data --out-dir reports --window-hours 36
"""

from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import orjson
from dateutil import parser as dtparse

# OpenAI SDK v1.x
from openai import OpenAI

# --------------------------- Config ---------------------------

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_ITEMS = 120  # cap incidents fed to the model for context control

DEFAULT_SYSTEM_PROMPT = """You are an OSINT analyst focused on drone incidents (smuggling, prison drops, arrests, crashes, terrorism, policy/security).
Objectives:
1) Cluster related incidents (same theme or geography).
2) Produce concise highlights with risk tags (e.g., prison, smuggling, airspace, critical-infrastructure, cross-border, wildfire, military, civilian-casualty, airport-disruption).
3) Normalize country names if obvious from title/description/source.
4) Score priority in [0,1] for policy/customer-facing relevance (0=low, 1=high).
Rules:
- Be conservative with inferences; if country is unclear, omit it.
- Prefer precision over verbosity; one sentence per highlight.
- No duplicate incident IDs in highlights.
- Output MUST validate the provided JSON Schema (when enforced).
"""

SCHEMA = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "date_utc": {"type": "string", "description": "ISO8601 UTC date for the brief"},
    "window_hours": {"type": "integer"},
    "totals": {
      "type": "object",
      "additionalProperties": False,
      "properties": {
        "total_incidents": {"type": "integer"},
        "by_category": {"type": "object", "additionalProperties": {"type": "integer"}},
        "by_query": {"type": "object", "additionalProperties": {"type": "integer"}}
      },
      "required": ["total_incidents","by_category","by_query"]
    },
    "clusters": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "topic": {"type": "string"},
          "rationale": {"type": "string"},
          "incident_ids": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["topic","incident_ids"]
      }
    },
    "highlights": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "id": {"type": "string"},
          "headline": {"type": "string"},
          "one_sentence": {"type": "string"},
          "risk_tags": {"type": "array", "items": {"type": "string"}},
          "countries": {"type": "array", "items": {"type": "string"}},
          "priority_score": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["id","headline","one_sentence","priority_score"]
      }
    },
    "draft_social": {
      "type": "object",
      "additionalProperties": False,
      "properties": {
        "tweet": {"type": "string"},
        "linkedin": {"type": "string"}
      }
    }
  },
  "required": ["date_utc","window_hours","totals","clusters","highlights"]
}

# --------------------------- Helpers ---------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def strip_json_fences(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        # remove ```json or ``` fences
        t = t.strip("`")
        # in case of ```json\n ... \n```
        t = t.replace("json\n", "", 1) if t.startswith("json\n") else t
    t = t.strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    # sometimes models prepend text; find first { ... last }
    first = t.find("{")
    last = t.rfind("}")
    return t[first:last+1] if first != -1 and last != -1 else t

def load_incidents(data_dir: Path) -> list[dict]:
    """Read every *.json under data_dir and merge by id/url, keeping newest by detected_at."""
    recs: dict[str, dict] = {}
    for p in sorted(data_dir.rglob("*.json")):
        try:
            with p.open("rb") as f:
                payload = orjson.loads(f.read())
            for it in payload.get("incidents", []):
                key = it.get("id") or it.get("url")
                if not key:
                    continue
                prev = recs.get(key)
                if not prev:
                    recs[key] = it
                else:
                    d_new = dtparse.isoparse(it.get("detected_at")) if it.get("detected_at") else None
                    d_old = dtparse.isoparse(prev.get("detected_at")) if prev.get("detected_at") else None
                    if d_new and (not d_old or d_new > d_old):
                        recs[key] = it
        except Exception as e:
            print(f"[warn] failed to parse {p}: {e}", file=sys.stderr)
    return list(recs.values())

def within_window(inc: dict, since: datetime) -> bool:
    for k in ("detected_at", "date"):
        v = inc.get(k)
        if v:
            try:
                t = dtparse.isoparse(v)
                return t >= since
            except Exception:
                continue
    return False

def to_minimal(inc: dict) -> dict:
    return {
        "id": inc.get("id"),
        "title": inc.get("title"),
        "url": inc.get("url"),
        "source": inc.get("source"),
        "date": inc.get("date"),
        "detected_at": inc.get("detected_at"),
        "description": inc.get("description"),
        "query": inc.get("query"),
        "category": inc.get("category"),
        "location": inc.get("location"),
    }

def render_markdown(data: dict, incidents_by_id: dict[str, dict]) -> str:
    date_str = datetime.now(timezone.utc).strftime("%B %d, %Y")
    totals = data.get("totals", {})
    lines = []
    lines.append(f"# Drone News Daily Brief — {date_str}\n")
    lines.append(f"_Window: last {data.get('window_hours', '?')} hours; {totals.get('total_incidents','?')} incidents._\n")
    if totals.get("by_category"):
        cats = " • ".join(f"{k}: {v}" for k,v in sorted(totals["by_category"].items()))
        lines.append(f"**By category:** {cats}")
    if totals.get("by_query"):
        qtop = sorted(totals["by_query"].items(), key=lambda kv: kv[1], reverse=True)[:5]
        lines.append("**Top queries:** " + ", ".join(f"{k} ({v})" for k,v in qtop))
    lines.append("")
    if data.get("clusters"):
        lines.append("## Clusters")
        for cl in data["clusters"]:
            topic = cl.get("topic","(untitled)")
            rationale = cl.get("rationale","")
            lines.append(f"**{topic}** — {rationale}")
            for iid in cl.get("incident_ids", [])[:8]:
                inc = incidents_by_id.get(iid)
                if not inc: 
                    continue
                lines.append(f"- [{inc.get('title')}]({inc.get('url')}) — {inc.get('source')}")
            lines.append("")
    if data.get("highlights"):
        lines.append("## Highlights")
        for h in data["highlights"][:10]:
            inc = incidents_by_id.get(h.get("id"))
            link = inc["url"] if inc else None
            head = h.get("headline","")
            one = h.get("one_sentence","")
            tags = ", ".join(h.get("risk_tags", [])) or "—"
            score = f"{h.get('priority_score', 0):.2f}"
            if link:
                lines.append(f"- **[{head}]({link})** — {one} _(score {score}; tags: {tags})_")
            else:
                lines.append(f"- **{head}** — {one} _(score {score}; tags: {tags})_")
    ds = data.get("draft_social", {})
    if ds.get("tweet") or ds.get("linkedin"):
        lines.append("\n---\n### Draft social")
        if ds.get("tweet"): lines.append(f"**Tweet**: {ds['tweet']}")
        if ds.get("linkedin"): lines.append(f"**LinkedIn**: {ds['linkedin']}")
    lines.append("")
    return "\n".join(lines)

# ---------------------- Model Invocation ----------------------

def call_model_with_fallback(client: OpenAI, model: str, system_prompt: str, user_prompt: str, json_schema: dict) -> dict:
    """
    Preferred: Responses API with JSON Schema structured outputs.
    Fallback:  Chat Completions with JSON mode (schema not enforced, but valid JSON).
    """
    # Try Responses API (newer SDKs)
    try:
        kwargs = {
            "model": model,
            # 'instructions' is the system-equivalent for Responses API
            "instructions": system_prompt,
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "DroneDailyBrief",
                    "schema": json_schema,
                    "strict": True,
                },
            },
        }
        resp = client.responses.create(**kwargs)
        text = getattr(resp, "output_text", None)
        if not text:
            # defensive extraction if SDK shape differs
            try:
                parts = []
                for block in getattr(resp, "output", []):
                    for c in getattr(block, "content", []):
                        if getattr(c, "type", "") == "output_text":
                            parts.append(getattr(c, "text"))
                text = "".join(parts) if parts else None
            except Exception:
                pass
        if not text:
            raise RuntimeError("Empty response from model.")
        cleaned = strip_json_fences(text)
        return json.loads(cleaned)
    except TypeError:
        # Older SDK path — Chat Completions JSON mode
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        cc_kwargs = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        cc = client.chat.completions.create(**cc_kwargs)
        text = cc.choices[0].message.content
        cleaned = strip_json_fences(text)
        return json.loads(cleaned)

# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory containing incident JSON files")
    ap.add_argument("--out-dir", required=True, help="Directory to write reports")
    ap.add_argument("--window-hours", type=int, default=36)
    ap.add_argument("--system-prompt", type=str, default="", help="Optional path to a system prompt .md")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    all_inc = load_incidents(data_dir)
    if not all_inc:
        print("No incidents found; nothing to do.")
        return

    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.window_hours)
    recent = [to_minimal(x) for x in all_inc if within_window(x, cutoff)]
    if not recent:
        print("No recent incidents in window; nothing to do.")
        return

    # dedupe & limit
    recent.sort(key=lambda r: r.get("detected_at") or r.get("date") or "", reverse=True)
    if len(recent) > MAX_ITEMS:
        recent = recent[:MAX_ITEMS]

    # lookup by id for rendering
    by_id = {x.get("id"): x for x in recent if x.get("id")}

    # Prepare prompts
    if args.system_prompt and Path(args.system_prompt).exists():
        system_prompt = Path(args.system_prompt).read_text(encoding="utf-8")
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    payload = {
        "window_hours": args.window_hours,
        "incidents": recent,
    }
    user_prompt = (
        "You will receive a list of recent drone-related incidents. "
        "Return a single JSON object matching the provided JSON Schema. "
        "Do not include any commentary outside of the JSON.\n\n"
        f"{orjson.dumps(payload).decode()}"
    )

    client = OpenAI()
    result = call_model_with_fallback(
        client=client,
        model=DEFAULT_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_schema=SCHEMA,
    )

    # Write outputs
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    json_path = out_dir / f"{today_utc}.json"
    md_path = out_dir / f"{today_utc}.md"

    with json_path.open("wb") as f:
        f.write(orjson.dumps(result, option=orjson.OPT_INDENT_2))

    md = render_markdown(result, by_id)
    md_path.write_text(md, encoding="utf-8")

    print(f"Wrote {md_path} and {json_path}")

if __name__ == "__main__":
    main()
