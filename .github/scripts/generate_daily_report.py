#!/usr/bin/env python3
"""
Generate a human-readable daily drone-incident report
(using the OpenAI ChatGPT API) and save it to reports/YYYY-MM-DD.md.

Expects:  OPENAI_API_KEY  in the environment
Outputs:  reports/<today>.md
Sets GITHUB_OUTPUT fields: report_created, report_path
"""

import json, os, sys, hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import openai
from dateutil import parser as dt

# ------------------------------------------------------------------ config --- #
INCIDENT_FILE = Path("data/incidents.json")
REPORT_DIR    = Path("reports")          # committed in repo
WINDOW_H      = 24                      # ‘daily’ window (hours)

# ------------------------------------------------------------------ helpers --- #
def load_recent_incidents():
    if not INCIDENT_FILE.exists():
        print("No incidents.json yet – skipping")
        return []

    with INCIDENT_FILE.open() as f:
        data = json.load(f)
    now  = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=WINDOW_H)

    fresh = [
        inc for inc in data["incidents"]
        if dt.isoparse(inc["detected_at"]) >= cutoff
    ]
    return fresh

def build_chat_prompt(incidents, report_date):
    """Return messages list for ChatCompletion"""
    # give ChatGPT a concise table of the incidents
    bullet_lines = []
    for inc in incidents:
        bullet_lines.append(
            f"- [{inc['category']}] {inc['title']} "
            f"({inc['source']}, {inc['location']})"
        )
    bullets = "\n".join(bullet_lines)

    system   = (
        "You are an aviation-security analyst.  "
        "Write a crisp, plain-English one-page daily brief for executives.  "
        "Group incidents by category, highlight notable events, and keep it under 400 words."
    )
    user     = (
        f"Date: {report_date}\n"
        f"Incidents ({len(incidents)}):\n{bullets}\n\n"
        "Write the daily drone-incident report."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

def call_chatgpt(messages):
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()

# ------------------------------------------------------------------- main ---- #
def main():
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    incidents = load_recent_incidents()
    if not incidents:
        # expose flag to the workflow and exit cleanly
        with open(os.environ.get("GITHUB_OUTPUT", "output.txt"), "a") as f:
            f.write("report_created=false\n")
        print("No incidents in window – no report created")
        return

    today = datetime.utcnow().strftime("%Y-%m-%d")
    prompts = build_chat_prompt(incidents, today)
    report_md = call_chatgpt(prompts)

    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / f"{today}.md"
    report_path.write_text(report_md, encoding="utf-8")

    with open(os.environ.get("GITHUB_OUTPUT", "output.txt"), "a") as f:
        f.write("report_created=true\n")
        f.write(f"report_path={report_path.as_posix()}\n")

    print(f"Wrote {report_path}")

if __name__ == "__main__":
    main()
