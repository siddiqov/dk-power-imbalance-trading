import requests
import json
import pandas as pd

url = "https://umm.nordpoolgroup.com/api/messages?marketType=Electricity&eventStatus=Active"
try:
    r = requests.get(url, timeout=10)
    data = r.json()
    items = data.get('items', [])
    print(f"Found {len(items)} active UMMs")
    for msg in items[:2]:
        print(msg.get('messageId'), msg.get('biddingAreas'), msg.get('eventType'))
except Exception as e:
    print("Error:", e)
