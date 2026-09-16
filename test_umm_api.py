import requests
url = 'https://umm.nordpoolgroup.com/api/messages?marketType=Electricity&eventStatus=Active'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'application/json'
}
response = requests.get(url, headers=headers, timeout=10)
print(response.status_code)
try:
    print(response.json())
except:
    print(response.text[:200])