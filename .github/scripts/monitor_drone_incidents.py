#!/usr/bin/env python3
"""
Drone Incident Monitor Script
Searches Google News RSS for drone-related incidents and keeps
data/incidents.json updated.  If new incidents are found it sets
GitHub-Actions outputs for downstream steps (release, commit, etc.).
"""

import json
import os
import sys
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import requests
import feedparser
from dateutil import parser as date_parser
from bs4 import BeautifulSoup


# --------------------------------------------------------------------------- #
#                              Configuration                                  #
# --------------------------------------------------------------------------- #

INCIDENT_FILE = Path("data/incidents.json")
SEARCH_QUERIES = [
    "drone incident",
    "drone arrest",
    "drone smuggling",
    "drone prison",
    "drone airport disruption",
    "drone crash",
    "drone illegal",
    "drone security breach",
    "drone terrorism",
    "unauthorized drone",
]


# --------------------------------------------------------------------------- #
#                              Main class                                     #
# --------------------------------------------------------------------------- #

class DroneIncidentMonitor:
    def __init__(self):
        self.incidents_file = INCIDENT_FILE
        self.existing_incidents = self.load_incidents()
        self.new_incidents: List[Dict] = []

    # ----------------------------- persistence ----------------------------- #

    def load_incidents(self) -> Dict:
        if self.incidents_file.exists():
            with self.incidents_file.open() as f:
                return json.load(f)
        return {"incidents": [], "last_updated": None, "total_count": 0}

    def save_incidents(self):
        self.existing_incidents["last_updated"] = datetime.now(
            timezone.utc
        ).isoformat()
        self.existing_incidents["total_count"] = len(
            self.existing_incidents["incidents"]
        )
        self.incidents_file.parent.mkdir(parents=True, exist_ok=True)
        with self.incidents_file.open("w") as f:
            json.dump(self.existing_incidents, f, indent=2, ensure_ascii=False)

    # ------------------------------ helpers -------------------------------- #

    @staticmethod
    def generate_incident_id(inc: Dict) -> str:
        key = f"{inc['title']}{inc['source']}{inc['date']}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def is_duplicate(self, inc_id: str) -> bool:
        return any(inc["id"] == inc_id for inc in self.existing_incidents["incidents"])

    # ------------------------------ parsing -------------------------------- #

    def search_google_news(self, query: str) -> List[Dict]:
        encoded = requests.utils.quote(query)
        rss_url = (
            f"https://news.google.com/rss/search?q={encoded}"
            "&hl=en-US&gl=US&ceid=US:en"
        )

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=48)
        incidents: List[Dict] = []

        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:10]:
                pub_date = date_parser.parse(entry.published)
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=timezone.utc)
                pub_date = pub_date.astimezone(timezone.utc)

                if pub_date < cutoff:
                    continue

                incident = {
                    "title": entry.title,
                    "url": entry.link,
                    "source": getattr(entry, "source", {}).get("title", "Unknown")
                    if hasattr(entry, "source")
                    else "Unknown",
                    "date": pub_date.isoformat(),
                    "description": BeautifulSoup(
                        entry.summary, "html.parser"
                    ).text[:500],
                    "query": query,
                    "detected_at": now.isoformat(),
                }
                full_text = incident["title"] + " " + incident["description"]
                incident["location"] = self.extract_location(full_text)
                incident["category"] = self.categorize_incident(full_text)
                incidents.append(incident)
        except Exception as exc:
            print(f"[WARN] error searching '{query}': {exc}", file=sys.stderr)

        return incidents

    # ----------------------- tiny NLP stubs -------------------------------- #

    @staticmethod
    def extract_location(txt: str) -> str:
        import re

        patterns = [
            r"in (\w+(?:\s+\w+)*),\s*([A-Z]{2})",
            r"at ([\w\s]+? (?:Airport|Prison|Jail))",
            r"near (\w+(?:\s+\w+)*)",
        ]
        for pat in patterns:
            m = re.search(pat, txt, re.IGNORECASE)
            if m:
                return m.group(0)
        return "Unknown"

    @staticmethod
    def categorize_incident(txt: str) -> str:
        txt = txt.lower()
        buckets = {
            "smuggling": ["smuggl", "contraband", "drug", "prison", "jail"],
            "airport": ["airport", "runway", "aviation", "faa"],
            "security": ["breach", "restrict", "unauthorized", "trespass"],
            "crash": ["crash", "collision", "accident"],
            "arrest": ["arrest", "detain", "custody", "charge"],
            "surveillance": ["spy", "surveillance", "privacy", "peeping"],
            "terrorism": ["terror", "threat", "weapon"],
            "interference": ["interfer", "disrupt", "jam"],
        }
        for cat, kws in buckets.items():
            if any(k in txt for k in kws):
                return cat
        return "other"

    # -------------------------- release notes ------------------------------ #

    def generate_release_notes(self) -> str:
        by_cat: Dict[str, List[Dict]] = {}
        for inc in self.new_incidents:
            by_cat.setdefault(inc["category"], []).append(inc)

        lines: List[str] = []
        for cat, incs in by_cat.items():
            lines.append(f"\n### {cat.title()} ({len(incs)} incidents)")
            for inc in incs[:3]:
                lines.extend(
                    [
                        f"- **{inc['title']}**",
                        f"  - Location: {inc['location']}",
                        f"  - Source: {inc['source']}",
                        f"  - [Read more]({inc['url']})",
                    ]
                )
        return "\n".join(lines)

    # ------------------------------ main ----------------------------------- #

    def run(self):
        print("📡  Drone incident monitor starting...\n")
        all_incidents: List[Dict] = []

        for q in SEARCH_QUERIES:
            print(f"🔍  {q}")
            all_incidents.extend(self.search_google_news(q))

        for inc in all_incidents:
            inc["id"] = self.generate_incident_id(inc)
            if not self.is_duplicate(inc["id"]):
                self.new_incidents.append(inc)
                self.existing_incidents["incidents"].insert(0, inc)

        # trim to last 1000
        self.existing_incidents["incidents"] = self.existing_incidents["incidents"][:1000]
        self.save_incidents()

        gha_out = os.environ.get("GITHUB_OUTPUT", "output.txt")
        now = datetime.now(timezone.utc)

        with open(gha_out, "a") as f:
            if self.new_incidents:
                f.write("new_incidents=true\n")
                f.write(f"incident_count={len(self.new_incidents)}\n")
                f.write(f"timestamp={now.strftime('%Y%m%d-%H%M%S')}\n")
                f.write(f"date={now.strftime('%B %d, %Y')}\n")

                locs = {
                    inc["location"]
                    for inc in self.new_incidents
                    if inc["location"] != "Unknown"
                }
                f.write(f"locations={', '.join(sorted(locs))}\n")

                release_notes = self.generate_release_notes()
                # <<< multi-line-safe GitHub Actions output >>>
                f.write("release_notes<<EOF\n")
                f.write(release_notes + "\n")
                f.write("EOF\n")
            else:
                f.write("new_incidents=false\n")

        print(
            f"\n✅  {len(self.new_incidents)} new incidents stored."
            if self.new_incidents
            else "\n✅  No new incidents found."
        )


# --------------------------------------------------------------------------- #
#                                   CLI                                       #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    DroneIncidentMonitor().run()
