#!/usr/bin/env python3
"""
Drone News V2.1.1 — reads incident JSON files, asks ChatGPT to cluster/score/summarize,
then writes:
  - reports/YYYY-MM-DD.md   (human-readable brief incl. 100–200 word summary)
  - reports/YYYY-MM-DD.json (structured output)

ENV:
  - OPENAI_API_KEY   (required)
  - OPENAI_MODEL     (optional; default 'gpt-4o-mini')

CLI:
  python .github/scripts/drone_v2_report.py --data-dir data --out-dir reports --window-hours 48 --system-prompt prompts/system_drone_analyst.md
"""

from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import orjson
from dateutil import parser as dtparse
from openai import OpenAI

# --------------------------- Config ---------------------------

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_ITEMS = 120  # cap incidents fed to the model for context control

# JSON Schema with REQUIRED narrative_summary (100–200 words requested in prompt)
SCHEMA = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "date_utc": {"type": "string", "description": "ISO8601 UTC date for the brief"},
    "window_hours": {"type": "integer"},
    "narrative_summary": {"type": "string", "description": "100–200 word prose summary of patterns over the lookback window"},
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
  "required": ["date_utc","window_hours","narrative_summary","totals","clusters","highlights"]
}

DEFAULT_SYSTEM_PROMPT = """You are an OSINT analyst focused on drone incidents (smuggling, prison drops, arrests, crashes, terrorism, policy/security).

Objectives:
1) Write a 100–200 word narrative_summary synthesizing the last N hours (N is provided as window_hours). Cover geographic spread, themes, and notable risks in plain English.
2) Compute totals by category and query.
3) Cluster related incidents and label each cluster.
4) Produce concise highlights with risk tags (e.g., prison, smuggling, airspace, critical-infrastructure, cross-border, wildfire, military, civilian-casualty, airport-disruption) and priority_score in [0,1].

Rules:
- Be conservative with inferences; if country is unclear, omit it.
- Prefer precision over verbosity; one sentence per highlight.
- No duplicate incident IDs in highlights.
- Output MUST validate the provided JSON Schema (when enforced).
- STRUCTURE: 'clusters' must be an array of objects (not strings). 'highlights' must be an array of objects with required fields.
"""

# --------------------------- Helpers ---------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def strip_json_fences(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        t = t.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    # best-effort: grab outermost JSON braces
    if t.startswith("{") and t.endswith("}"):
        return t
    first, last = t.find("{"), t.rfind("}")
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

def compute_totals(recent: list[dict]) -> dict:
    by_cat, by_q = {}, {}
    for r in recent:
        c = (r.get("category") or "uncategorized").strip() or "uncategorized"
        q = (r.get("query") or "unknown").strip() or "unknown"
        by_cat[c] = by_cat.get(c, 0) + 1
        by_q[q] = by_q.get(q, 0) + 1
    return {
        "total_incidents": len(recent),
        "by_category": dict(sorted(by_cat.items())),
        "by_query": dict(sorted(by_q.items())),
    }

def sanitize_result_structures(res: dict, incidents_by_id: dict[str, dict]) -> dict:
    """Normalize clusters/highlights to avoid attribute errors even with malformed model output."""
    out = dict(res)

    # clusters → list of {topic, rationale, incident_ids}
    raw_clusters = out.get("clusters", [])
    norm_clusters = []
    if isinstance(raw_clusters, list):
        for cl in raw_clusters:
            if isinstance(cl, dict):
                topic = str(cl.get("topic", "(untitled)"))
                rationale = str(cl.get("rationale", ""))
                ids = cl.get("incident_ids") or []
                if not isinstance(ids, (list, tuple)):
                    ids = []
                ids = [str(x) for x in ids if isinstance(x, (str, int))]
                norm_clusters.append({"topic": topic, "rationale": rationale, "incident_ids": ids})
            elif isinstance(cl, (list, tuple)):
                ids = [str(x) for x in cl if isinstance(x, (str, int))]
                norm_clusters.append({"topic": "(unnamed cluster)", "rationale": "", "incident_ids": ids})
            elif isinstance(cl, str):
                norm_clusters.append({"topic": cl, "rationale": "", "incident_ids": []})
    elif isinstance(raw_clusters, dict):
        # Rare: a mapping of topic->ids
        for topic, ids in raw_clusters.items():
            if isinstance(ids, (list, tuple)):
                ids = [str(x) for x in ids if isinstance(x, (str, int))]
            else:
                ids = []
            norm_clusters.append({"topic": str(topic), "rationale": "", "incident_ids": ids})
    else:
        norm_clusters = []

    out["clusters"] = norm_clusters

    # highlights → list of full objects
    raw_high = out.get("highlights", [])
    norm_high = []
    if isinstance(raw_high, list):
        for h in raw_high:
            if isinstance(h, dict):
                hid = str(h.get("id") or "")
                headline = str(h.get("headline") or incidents_by_id.get(hid, {}).get("title") or "").strip()
                one = str(h.get("one_sentence") or "").strip()
                risk = h.get("risk_tags");  risk = risk if isinstance(risk, list) else []
                countries = h.get("countries"); countries = countries if isinstance(countries, list) else []
                try:
                    score = float(h.get("priority_score"))
                except Exception:
                    score = 0.5
                norm_high.append({
                    "id": hid, "headline": headline, "one_sentence": one,
                    "risk_tags": risk, "countries": countries, "priority_score": score
                })
            elif isinstance(h, str):
                norm_high.append({
                    "id": "", "headline": h[:100], "one_sentence": h,
                    "risk_tags": [], "countries": [], "priority_score": 0.5
                })
    out["highlights"] = norm_high

    return out

def fill_missing_fields(result: dict, recent: list[dict], window_hours: int) -> dict:
    """Ensure required fields exist; compute totals if missing to avoid '?' in output."""
    res = dict(result)
    if "window_hours" not in res or not isinstance(res["window_hours"], int):
        res["window_hours"] = window_hours
    if "totals" not in res or not isinstance(res["totals"], dict):
        res["totals"] = compute_totals(recent)
    else:
        t = res["totals"]
        if "total_incidents" not in t or not isinstance(t["total_incidents"], int):
            t["total_incidents"] = len(recent)
        if "by_category" not in t or not isinstance(t["by_category"], dict):
            t["by_category"] = compute_totals(recent)["by_category"]
        if "by_query" not in t or not isinstance(t["by_query"], dict):
            t["by_query"] = compute_totals(recent)["by_query"]
    if "narrative_summary" not in res or not isinstance(res["narrative_summary"], str):
        res["narrative_summary"] = (
            f"In the past {res['window_hours']} hours, {res['totals']['total_incidents']} drone-related "
            f"incidents were recorded across multiple categories. See clusters and highlights below."
        )
    if "date_utc" not in res or not isinstance(res["date_utc"], str):
        res["date_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # ensure containers exist
    if "clusters" not in res or not isinstance(res["clusters"], list):
        res["clusters"] = []
    if "highlights" not in res or not isinstance(res["highlights"], list):
        res["highlights"] = []
    if "draft_social" not in res or not isinstance(res["draft_social"], dict):
        res["draft_social"] = {"tweet": "", "linkedin": ""}
    return res

def render_markdown(data: dict, incidents_by_id: dict[str, dict]) -> str:
    date_str = datetime.now(timezone.utc).strftime("%B %d, %Y")
    totals = data.get("totals", {})
    total_count = totals.get("total_incidents", len(incidents_by_id))
    lines = []
    lines.append(f"# Drone News 48h Brief — {date_str}\n")
    lines.append(f"_Window: last {data.get('window_hours', '?')} hours; {total_count} incidents._\n")
    # Narrative summary first (100–200 words)
    if data.get("narrative_summary"):
        lines.append("## Summary")
        lines.append(data["narrative_summary"].strip())
        lines.append("")
    # Totals
    if isinstance(totals.get("by_category"), dict) and totals["by_category"]:
        cats = " • ".join(f"{k}: {v}" for k,v in sorted(totals["by_category"].items()))
        lines.append(f"**By category:** {cats}")
    if isinstance(totals.get("by_query"), dict) and totals["by_query"]:
        qtop = sorted(totals["by_query"].items(), key=lambda kv: kv[1], reverse=True)[:5]
        lines.append("**Top queries:** " + ", ".join(f"{k} ({v})" for k,v in qtop))
    lines.append("")
    # Clusters
    clusters = data.get("clusters") or []
    if clusters:
        lines.append("## Clusters")
        for cl in clusters:
            if not isinstance(cl, dict):
                continue
            topic = cl.get("topic","(untitled)")
            rationale = cl.get("rationale","")
            lines.append(f"**{topic}** — {rationale}")
            for iid in (cl.get("incident_ids") or [])[:8]:
                inc = incidents_by_id.get(iid)
                if not inc:
                    continue
                lines.append(f"- [{inc.get('title')}]({inc.get('url')}) — {inc.get('source')}")
            lines.append("")
    # Highlights
    highlights = data.get("highlights") or []
    if highlights:
        lines.append("## Highlights")
        for h in highlights[:10]:
            if not isinstance(h, dict):
                continue
            inc = incidents_by_id.get(h.get("id"))
            link = inc["url"] if inc else None
            head = h.get("headline","")
            one = h.get("one_sentence","")
            tags = ", ".join(h.get("risk_tags", [])) or "—"
            try:
                score = f"{float(h.get('priority_score', 0)):.2f}"
            except Exception:
                score = "0.00"
            if link:
                lines.append(f"- **[{head}]({link})** — {one} _(score {score}; tags: {tags})_")
            else:
                lines.append(f"- **{head}** — {one} _(score {score}; tags: {tags})_")
    # Draft social
    ds = data.get("draft_social", {})
    if isinstance(ds, dict) and (ds.get("tweet") or ds.get("linkedin")):
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
            "instructions": system_prompt,  # system-equivalent for Responses API
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
            # defensive extraction
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
    ap.add_argument("--window-hours", type=int, default=48, help="Lookback window in hours (default 48)")
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
        # Still write an empty brief for traceability
        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        md_path = out_dir / f"{today_utc}.md"
        json_path = out_dir / f"{today_utc}.json"
        empty = {
            "date_utc": today_utc,
            "window_hours": args.window_hours,
            "narrative_summary": f"No incidents detected in the last {args.window_hours} hours.",
            "totals": {"total_incidents": 0, "by_category": {}, "by_query": {}},
            "clusters": [],
            "highlights": [],
            "draft_social": {"tweet": "", "linkedin": ""},
        }
        md = render_markdown(empty, {})
        md_path.write_text(md, encoding="utf-8")
        with json_path.open("wb") as f:
            f.write(orjson.dumps(empty, option=orjson.OPT_INDENT_2))
        print(f"Wrote {md_path} and {json_path}")
        return

    # dedupe & limit (already merged; now sort newest first)
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

    # Normalize/defend against malformed shapes from older SDK fallback
    result = sanitize_result_structures(result, by_id)
    # Ensure totals/fields exist so we never print '?' in the markdown.
    result = fill_missing_fields(result, recent, args.window_hours)

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
