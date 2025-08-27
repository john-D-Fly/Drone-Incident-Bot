#!/usr/bin/env python3
"""
Drone News V2 — reads incident JSON files, asks ChatGPT to cluster/score/summarize,
then writes:
  - reports/YYYY-MM-DD.md  (human brief)
  - reports/YYYY-MM-DD.json (structured output)
Requires:
  - OPENAI_API_KEY secret
Optional:
  - OPENAI_MODEL env (defaults to 'o4-mini')
Usage:
  python scripts/drone_v2_report.py --data-dir data --out-dir reports --window-hours 36
"""

from __future__ import annotations
import argparse, os, json, sys, glob
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dateutil import parser as dtparse
import orjson

# OpenAI Responses API
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from env

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "o4-mini")  # low-cost, solid reasoning

def load_incidents(data_dir: Path) -> list[dict]:
    recs: dict[str, dict] = {}
    for p in sorted(data_dir.rglob("*.json")):
        try:
            with p.open("rb") as f:
                payload = orjson.loads(f.read())
            items = payload.get("incidents", [])
            for it in items:
                key = it.get("id") or it.get("url")
                if not key: 
                    continue
                # prefer latest duplicate by detected_at
                prev = recs.get(key)
                if not prev:
                    recs[key] = it
                else:
                    d_new = dtparse.isoparse(it.get("detected_at")) if it.get("detected_at") else None
                    d_old = dtparse.isoparse(prev.get("detected_at")) if prev.get("detected_at") else None
                    if d_new and (not d_old or d_new > d_old):
                        recs[key] = it
        except Exception as e:
            print(f"[warn] failed {p}: {e}", file=sys.stderr)
    return list(recs.values())

def within_window(inc: dict, since: datetime) -> bool:
    # Prefer detected_at, fallback to date
    for k in ("detected_at", "date"):
        v = inc.get(k)
        if v:
            try:
                t = dtparse.isoparse(v)
                return t >= since
            except Exception:
                pass
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

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def call_model(incidents: list[dict], window_hours: int, system_prompt_path: Path | None) -> dict:
    # System prompt
    if system_prompt_path and system_prompt_path.exists():
        system_prompt = system_prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # JSON Schema the model must return (compact but expressive)
    schema = {
      "name": "DroneDailyBrief",
      "schema": {
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
      },
      "strict": True
    }

    # Compact incident list to feed the model
    payload = {
        "window_hours": window_hours,
        "incidents": incidents,
    }
    input_text = (
        "You will receive a list of recent drone-related incidents. "
        "Return a single JSON object matching the provided JSON Schema. "
        "Do not include any commentary outside of the JSON.\n\n"
        f"{orjson.dumps(payload).decode()}"
    )

    resp = client.responses.create(
        model=DEFAULT_MODEL,
        input=input_text,
        response_format={"type":"json_schema", "json_schema": schema},
        # Give the model a role & rules
        system=system_prompt
    )

    # Robust extraction of the JSON string
    raw = None
    try:
        raw = resp.output_text  # preferred helper
    except Exception:
        # Fallback: walk the content structure if needed
        try:
            chunks = []
            for block in getattr(resp, "output", []):
                for c in getattr(block, "content", []):
                    t = getattr(c, "text", None)
                    if t: chunks.append(t)
            raw = "".join(chunks) if chunks else None
        except Exception:
            pass

    if not raw:
        raise RuntimeError("No text content returned from model.")

    try:
        return json.loads(raw)
    except Exception as e:
        # last resort: strip code fences if any
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned)

def render_markdown(data: dict, incidents_by_id: dict[str, dict]) -> str:
    # Build a tidy brief
    date_str = datetime.now(timezone.utc).strftime("%B %d, %Y")
    totals = data["totals"]
    lines = []
    lines.append(f"# Drone News Daily Brief — {date_str}\n")
    lines.append(f"_Window: last {data['window_hours']} hours; {totals['total_incidents']} incidents._\n")
    # Totals
    if totals.get("by_category"):
        cats = " • ".join(f"{k}: {v}" for k,v in sorted(totals["by_category"].items()))
        lines.append(f"**By category:** {cats}")
    if totals.get("by_query"):
        qtop = sorted(totals["by_query"].items(), key=lambda kv: kv[1], reverse=True)[:5]
        lines.append("**Top queries:** " + ", ".join(f"{k} ({v})" for k,v in qtop))
    lines.append("")
    # Clusters
    if data.get("clusters"):
        lines.append("## Clusters")
        for cl in data["clusters"]:
            lines.append(f"**{cl['topic']}** — {cl.get('rationale','')}")
            for iid in cl.get("incident_ids", [])[:8]:
                inc = incidents_by_id.get(iid)
                if not inc: continue
                lines.append(f"- [{inc.get('title')}]({inc.get('url')}) — {inc.get('source')}")
            lines.append("")
    # Highlights
    if data.get("highlights"):
        lines.append("## Highlights")
        for h in data["highlights"][:10]:
            inc = incidents_by_id.get(h["id"])
            link = inc["url"] if inc else None
            head = h["headline"]
            one = h["one_sentence"]
            tags = ", ".join(h.get("risk_tags", [])) or "—"
            score = f"{h['priority_score']:.2f}"
            if link:
                lines.append(f"- **[{head}]({link})** — {one} _(score {score}; tags: {tags})_")
            else:
                lines.append(f"- **{head}** — {one} _(score {score}; tags: {tags})_")
    # Draft social
    ds = data.get("draft_social", {})
    if ds.get("tweet") or ds.get("linkedin"):
        lines.append("\n---\n### Draft social")
        if ds.get("tweet"): lines.append(f"**Tweet**: {ds['tweet']}")
        if ds.get("linkedin"): lines.append(f"**LinkedIn**: {ds['linkedin']}")
    lines.append("")
    return "\n".join(lines)

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
- Output MUST validate the provided JSON Schema.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window-hours", type=int, default=36)
    ap.add_argument("--system-prompt", type=str, default="")
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

    # Build lookup by id for rendering & links
    by_id = {x.get("id"): x for x in recent if x.get("id")}

    # Truncate very large batches to keep context sane
    MAX_ITEMS = 120
    if len(recent) > MAX_ITEMS:
        # Keep the newest N by detected_at
        recent.sort(key=lambda r: r.get("detected_at") or r.get("date") or "", reverse=True)
        recent = recent[:MAX_ITEMS]

    result = call_model(
        incidents=recent,
        window_hours=args.window_hours,
        system_prompt_path=Path(args.system_prompt) if args.system_prompt else None,
    )

    # Write JSON + Markdown
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
