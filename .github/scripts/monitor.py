#!/usr/bin/env python3
"""
Drone Incident Monitor Script
Searches for drone-related incidents and updates the incidents database
"""

import json
import os
import sys
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Set
import requests
import feedparser
from dateutil import parser as date_parser
from bs4 import BeautifulSoup

class DroneIncidentMonitor:
    def __init__(self):
        self.incidents_file = 'data/incidents.json'
        self.existing_incidents = self.load_incidents()
        self.new_incidents = []
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
            "unauthorized drone"
        ]
        
    def load_incidents(self) -> Dict:
        """Load existing incidents from JSON file"""
        if os.path.exists(self.incidents_file):
            with open(self.incidents_file, 'r') as f:
                return json.load(f)
        return {"incidents": [], "last_updated": None, "total_count": 0}
    
    def save_incidents(self):
        """Save incidents to JSON file"""
        os.makedirs(os.path.dirname(self.incidents_file), exist_ok=True)
        self.existing_incidents['last_updated'] = datetime.now().isoformat()
        self.existing_incidents['total_count'] = len(self.existing_incidents['incidents'])
        
        with open(self.incidents_file, 'w') as f:
            json.dump(self.existing_incidents, f, indent=2, ensure_ascii=False)
    
    def generate_incident_id(self, incident: Dict) -> str:
        """Generate unique ID for incident based on content"""
        content = f"{incident['title']}{incident['source']}{incident['date']}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def is_duplicate(self, incident_id: str) -> bool:
        """Check if incident already exists"""
        existing_ids = {inc['id'] for inc in self.existing_incidents['incidents']}
        return incident_id in existing_ids
    
    def search_google_news(self, query: str) -> List[Dict]:
        """Search Google News RSS for drone incidents"""
        incidents = []
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:10]:  # Limit to 10 results per query
                # Parse date
                pub_date = date_parser.parse(entry.published)
                
                # Only include incidents from last 48 hours
                if pub_date < datetime.now() - timedelta(hours=48):
                    continue
                
                incident = {
                    'title': entry.title,
                    'url': entry.link,
                    'source': entry.source.title if hasattr(entry, 'source') else 'Unknown',
                    'date': pub_date.isoformat(),
                    'description': BeautifulSoup(entry.summary, 'html.parser').text[:500],
                    'query': query,
                    'detected_at': datetime.now().isoformat()
                }
                
                # Extract location if possible
                incident['location'] = self.extract_location(incident['title'] + ' ' + incident['description'])
                
                # Categorize incident
                incident['category'] = self.categorize_incident(incident['title'] + ' ' + incident['description'])
                
                incidents.append(incident)
                
        except Exception as e:
            print(f"Error searching for '{query}': {e}")
        
        return incidents
    
    def extract_location(self, text: str) -> str:
        """Extract location from text using simple pattern matching"""
        # This is a simplified version - in production, use NLP/NER
        import re
        
        # Common patterns for locations
        patterns = [
            r'in (\w+(?:\s+\w+)*), (\w{2})',  # City, State
            r'at (\w+(?:\s+\w+)*) (?:Airport|Prison|Jail)',  # Named facilities
            r'near (\w+(?:\s+\w+)*)',  # Near location
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return "Unknown"
    
    def categorize_incident(self, text: str) -> str:
        """Categorize incident based on keywords"""
        text_lower = text.lower()
        
        categories = {
            'smuggling': ['smuggl', 'contraband', 'drug', 'prison', 'jail'],
            'airport': ['airport', 'runway', 'aviation', 'faa'],
            'security': ['breach', 'restrict', 'unauthorized', 'trespass'],
            'crash': ['crash', 'collision', 'accident'],
            'arrest': ['arrest', 'detain', 'custody', 'charge'],
            'surveillance': ['spy', 'surveillance', 'privacy', 'peeping'],
            'terrorism': ['terror', 'threat', 'weapon'],
            'interference': ['interfer', 'disrupt', 'jam']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def run_monitor(self):
        """Main monitoring function"""
        print("Starting drone incident monitor...")
        
        all_incidents = []
        
        # Search for each query
        for query in self.search_queries:
            print(f"Searching for: {query}")
            incidents = self.search_google_news(query)
            all_incidents.extend(incidents)
        
        # Process and deduplicate incidents
        for incident in all_incidents:
            incident['id'] = self.generate_incident_id(incident)
            
            if not self.is_duplicate(incident['id']):
                self.new_incidents.append(incident)
                self.existing_incidents['incidents'].insert(0, incident)  # Add to beginning
        
        # Keep only last 1000 incidents
        self.existing_incidents['incidents'] = self.existing_incidents['incidents'][:1000]
        
        # Save updated incidents
        self.save_incidents()
        
        # Generate output for GitHub Actions
        if self.new_incidents:
            print(f"::set-output name=new_incidents::true")
            print(f"::set-output name=incident_count::{len(self.new_incidents)}")
            print(f"::set-output name=timestamp::{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            print(f"::set-output name=date::{datetime.now().strftime('%B %d, %Y')}")
            
            # Get unique locations
            locations = set(inc['location'] for inc in self.new_incidents if inc['location'] != 'Unknown')
            print(f"::set-output name=locations::{', '.join(locations)}")
            
            # Generate release notes
            release_notes = self.generate_release_notes()
            # Escape newlines for GitHub Actions
            release_notes_escaped = release_notes.replace('\n', '%0A')
            print(f"::set-output name=release_notes::{release_notes_escaped}")
        else:
            print(f"::set-output name=new_incidents::false")
            print("No new incidents found")
    
    def generate_release_notes(self) -> str:
        """Generate formatted release notes"""
        notes = []
        
        # Group by category
        categories = {}
        for incident in self.new_incidents:
            cat = incident['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(incident)
        
        for category, incidents in categories.items():
            notes.append(f"\n### {category.title()} ({len(incidents)} incidents)")
            for inc in incidents[:3]:  # Show max 3 per category
                notes.append(f"- **{inc['title']}**")
                notes.append(f"  - Location: {inc['location']}")
                notes.append(f"  - Source: {inc['source']}")
                notes.append(f"  - [Read more]({inc['url']})")
        
        return '\n'.join(notes)

if __name__ == "__main__":
    monitor = DroneIncidentMonitor()
    monitor.run_monitor()
