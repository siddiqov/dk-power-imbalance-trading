import requests
import pandas as pd
from datetime import datetime, timezone
import logging
import random
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/119.0"
]

# AREA EIC -> short name mapping for filtering
AREA_EIC_TO_SHORT = {
    "10YDK-1--------W": "DK1",
    "10YDK-2--------M": "DK2",
}

def fetch_live_umms(areas=None):
    """
    Fetches live Urgent Market Messages (UMMs) from Nord Pool's real public API.
    Returns the total MW capacity currently offline for the given areas.
    
    Uses the correct endpoint: https://ummapi.nordpoolgroup.com/messages
    """
    if areas is None:
        areas = ["DK1", "DK2"]
    
    url = "https://ummapi.nordpoolgroup.com/messages"
    
    # Anti-Bot: Random jitter to avoid cron-aligned ping spikes
    time.sleep(random.uniform(1.0, 4.5))
    
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            total_mw_offline = 0.0
            now = datetime.now(timezone.utc)
            
            items = data.get("items", [])
            for item in items:
                # Check production units for relevant areas
                for pu in item.get("productionUnits", []):
                    area_name = pu.get("areaName", "")
                    area_eic = pu.get("areaEic", "")
                    short_name = AREA_EIC_TO_SHORT.get(area_eic, area_name)
                    
                    if short_name not in areas:
                        continue
                    
                    # Sum unavailable capacity from active time periods
                    for tp in pu.get("timePeriods", []):
                        try:
                            start = datetime.fromisoformat(tp["eventStart"].replace("Z", "+00:00"))
                            stop = datetime.fromisoformat(tp["eventStop"].replace("Z", "+00:00"))
                            if start <= now <= stop:
                                cap = float(tp.get("unavailableCapacity", 0))
                                total_mw_offline += cap
                        except (ValueError, TypeError, KeyError):
                            pass
                
                # Check transmission units too (cable outages)
                for tu in item.get("transmissionUnits", []):
                    area_name = tu.get("areaName", "")
                    area_eic = tu.get("areaEic", "")
                    short_name = AREA_EIC_TO_SHORT.get(area_eic, area_name)
                    
                    if short_name not in areas:
                        continue
                    
                    for tp in tu.get("timePeriods", []):
                        try:
                            start = datetime.fromisoformat(tp["eventStart"].replace("Z", "+00:00"))
                            stop = datetime.fromisoformat(tp["eventStop"].replace("Z", "+00:00"))
                            if start <= now <= stop:
                                cap = float(tp.get("unavailableCapacity", 0))
                                total_mw_offline += cap
                        except (ValueError, TypeError, KeyError):
                            pass
            
            logger.info(f"UMM: {total_mw_offline:.0f} MW offline in {areas} (from {len(items)} messages)")
            return total_mw_offline
            
        else:
            logger.warning(f"Failed to fetch UMMs. Status code: {response.status_code}")
            return 0.0
            
    except Exception as e:
        logger.error(f"Error fetching UMMs: {e}")
        return 0.0

if __name__ == "__main__":
    mw_offline = fetch_live_umms(["DK1", "DK2"])
    print(f"Currently Offline Capacity in DK: {mw_offline:.0f} MW")