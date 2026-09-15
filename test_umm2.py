import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
try:
    r = requests.get("https://umm.nordpoolgroup.com/api/messages", headers=headers, timeout=10)
    print(r.status_code)
    print(r.text[:200])
except Exception as e:
    print(e)
