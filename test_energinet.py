import requests
url = 'https://api.energidataservice.dk/dataset/Forecasts_Hour?limit=2'
try:
    r = requests.get(url, timeout=5)
    print("Status:", r.status_code)
    print("Response:", r.text[:200])
except Exception as e:
    print(e)