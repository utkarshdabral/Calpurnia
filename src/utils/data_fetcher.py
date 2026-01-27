import requests
import os
from typing import Optional, List

def fetch_tle(catalog_id: int) -> Optional[str]:
    """Fetches TLE data and saves it to a pre-existing data directory."""
    
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={catalog_id}&FORMAT=TLE"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.text.strip()
        if not data:
            return None

        file_path = os.path.join("data", f"tle_{catalog_id}.txt")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(data)
            print(f"Successfully downloaded TLE for satellite:{catalog_id}")
        return data

    except requests.RequestException:
        return None

def fetch_multiple_tles(catalog_ids: List[int]) -> List[Optional[str]]:
    """Fetch TLEs for multiple satellites."""
    tles = []
    for cid in catalog_ids:
        tle = fetch_tle(cid)
        tles.append(tle)
    return tles

def get_active_satellites(limit: int = 10) -> List[int]:
    """Get list of active satellite NORAD IDs from Celestrak."""
    # For demo, return some known active satellites
    # In production, scrape from https://celestrak.org/NORAD/elements/active.txt
    active_ids = [
        25544,  # ISS
        44713,  # Starlink-24
        44714,  # Starlink-25
        44715,  # Starlink-26
        44716,  # Starlink-27
        44717,  # Starlink-28
        44718,  # Starlink-29
        44719,  # Starlink-30
        44720,  # Starlink-31
        44721,  # Starlink-32
    ]
    return active_ids[:limit]

if __name__ == "__main__":
    # Fetch ISS and some Starlinks
    ids = [25544, 44713, 44714, 44715]
    fetch_multiple_tles(ids)