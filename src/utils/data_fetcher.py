import requests
import os
from typing import Optional

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

if __name__ == "__main__":
    fetch_tle(25544)