#!/usr/bin/env python3
"""
Drone Incident Monitor Script
Searches for drone-related incidents and updates the incidents database
"""

import json
import os
import sys
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import requests
import feedparser
from dateutil import parser as date_parser
from bs4 import BeautifulSoup


class DroneIncidentMonitor:
    def __init__(self):
        self.incidents_file = 'data/incidents.json'
        self.existing_incidents = self.load_incidents()
        self.new_incidents: List[Dict] = []
        self.search_queries = [
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

    # --------------------------------------------------------------------- #
    # Utility / persistence helpers
    # --------------------------------------------------------------------- #

    def load_incidents(self) -> Dict:
        """Load existing incidents from JSON file (create skeleton if missing)."""
        if os.path.exists(self.incidents_file):
            with open(self.incidents_file, "r") as f:
                return json.load(f)
        return {"incidents": [], "last_updated": None, "total_count": 0}

    def save_incidents(self):
        """Persist incidents list to disk (update timestamp & total)."""
        os.makedirs(os.path.dirname(self.incidents_file), exist_ok=True)
        self.existing_incidents["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.existing_incidents["total_count"] = len(
            self.existing_incidents["incidents"]
        )

        with open(self.incidents_file, "w") as f:
            json.dump(self.existing_incidents, f, indent=2, ensure_ascii=False)

    def generate_incident_id(self, incident: Dict) -> str:
        """Generate a stable hash ID from title + source + date."""
        content = f"{incident['title']}{incident['source']}{incident['date']}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def is_duplicate(self, incident_id: str) -> bool:
        """True if we already stored this incident ID."""
        existing_ids = {inc["id"] for inc in self.existing_incidents["incidents"]}
        return incident_id in existing_ids

    # --------------------------------------------------------------------- #
    # Feeds / parsing
    # --------------------------------------------------------------------- #

    def search_google_news(self, query: str) -> List[Dict]:
        """Return a list of incident dicts for a single Google-News query."""
        incidents: List[Dict] = []

        encoded_query = requests.utils.quote(query)
        rss_url = (
            f"https://news.google.com/rss/search?q={encoded_query}"
            "&hl=en-US&gl=US&ceid=US:en"
        )

        try:
            feed = feedparser.parse(rss_url)
            now = datetime.now(timezone.utc)

            for entry in feed.entries[:10]:  # at most 10 per query
                # -- Parse and normalise publication date ------------------- #
                pub_date = date_parser.parse(entry.published)
                if pub_date.tzinfo is None:  # make it aware if feed omitted tz
                    pub_date = pub_date.replace(tzinfo=timezone.utc)
                pub_date = pub_date.astimezone(timezone.utc)

                # Only keep items <= 48 h old
                if pub_date < now - timedelta(hours=48):
                    continue

                incident = {
                    "title": entry.title,
                    "url": entry.link,
                    "source": getattr(entry, "source", {}).get("title", "Unknown")
                    if hasattr(entry, "source")
                    else "Unknown",
                    "date": pub_date.isoformat(),
                    "description": BeautifulSoup(entry.summary, "html.parser").text[:500],
                    "query": query,
                    "detected_at": now.isoformat(),
                }

                # Basic NLP stubs
                incident["location"] = self.extract_location(
                    incident["title"] + " " + incident["description"]
                )
                incident["category"] = self.categorize_incident(
                    incident["title"] + " " + incident["description"]
                )

                incidents.append(incident)

        except Exception as e:
            print(f"Error searching for '{query}': {e}", file=sys.stderr)

        return incidents

    # --------------------------------------------------------------------- #
    # Very-lightweight location / category helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def extract_location(text: str) -> str:
        """Return a rough location string if we can spot one."""
        import re

        patterns = [
            r"in (\w+(?:\s+\w+)*),\s*([A-Z]{2})",  # City, ST
            r"at ([\w\s]+? (?:Airport|Prison|Jail))",
            r"near (\w+(?:\s+\w+)*)",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                return m.group(0)
        return "Unknown"

    @staticmethod
    def categorize_incident(text: str) -> str:
        """Map free-text to a coarse incident category."""
        text_l = text.lower()
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
            if any(kw in text_l for kw in kws):
                return cat
        return "other"

    # --------------------------------------------------------------------- #
    # Release-note generation
    # --------------------------------------------------------------------- #

    def generate_release_notes(self) -> str:
        """Produce a GitHub-flavoured Markdown changelog snippet."""
        by_cat: Dict[str, List[Dict]] = {}
        for inc in self.new_incidents:
            by_cat.setdefault(inc["category"], []).append(inc)

        lines: List[str] = []
        for cat, incs in by_cat.items():
            lines.append(f"\n### {cat.title()} ({len(incs)} incidents)")
            for inc in incs[:3]:  # max 3 per cat
                lines.extend(
                    [
                        f"- **{inc['title']}**",
                        f"  - Location: {inc['location']}",
                        f"  - Source: {inc['source']}",
                        f"  - [Read more]({inc['url']})",
                    ]
                )
        return "\n".join(lines)

    # --------------------------------------------------------------------- #
    # Main entry-point
    # --------------------------------------------------------------------- #

    def run_monitor(self):
        print("Starting drone incident monitor…\n")
        all_incidents: List[Dict] = []

        # --- query each term ------------------------------------------------ #
        for query in self.search_queries:
            print(f"Searching for: {query}")
            all_incidents.extend(self.search_google_news(query))

        # --- deduplicate & store ------------------------------------------- #
        for inc in all_incidents:
            inc["id"] = self.generate_incident_id(inc)
            if not self.is_duplicate(inc["id"]):
                self.new_incidents.append(inc)
                self.existing_incidents["incidents"].insert(0, inc)

        # keep last 1000
        self.existing_incidents["incidents"] = self.existing_incidents["incidents"][:1000]
        self.save_incidents()

        # --- GitHub Actions outputs ---------------------------------------- #
        gha_output = os.environ.get("GITHUB_OUTPUT", "output.txt")
        now = datetime.now(timezone.utc)

        if self.new_incidents:
            with open(gha_output, "a") as f:
                f.write("new_incidents=true\n")
                f.write(f"incident_count={len(self.new_incidents)}\n")
                f.write(f"timestamp={now.strftime('%Y%m%d-%H%M%S')}\n")
                f.write(f"date={now.strftime('%B %d, %Y')}\n")

                locations = {
                    inc["location"] for inc in self.new_incidents if inc["location"] != "Unknown"
                }
                f.write(f"locations={', '.join(sorted(locations))}\n")

                rel_notes = self.generate_release_notes().replace("\n", "%0A")
                f.write(f"release_notes={rel_notes}\n")

            print(f"\nFound {len(self.new_incidents)} new incidents.")
        else:
            with open(gha_output, "a") as f:
                f.write("new_incidents=false\n")
            print("\nNo new incidents found.")


# ------------------------------------------------------------------------- #
# CLI
# ------------------------------------------------------------------------- #

if __name__ == "__main__":
    DroneIncidentMonitor().run_monitor()
