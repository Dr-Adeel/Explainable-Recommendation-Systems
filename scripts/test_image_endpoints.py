import requests
import sys

API = "http://127.0.0.1:8001"

TEST_IDS = [1,2,3,4,5,9,10,11,13,16]

def check_health():
    try:
        r = requests.get(f"{API}/health", timeout=5)
        r.raise_for_status()
        print("health:", r.json())
    except Exception as e:
        print("Health check failed:", e)
        sys.exit(2)

def test_images():
    for iid in TEST_IDS:
        url = f"{API}/amazon/image/{iid}"
        try:
            r = requests.get(url, timeout=10)
            print(f"item {iid}: status={r.status_code}", end="")
            if r.status_code == 200:
                ct = r.headers.get('content-type')
                print(f", content-type={ct}, bytes={len(r.content)}")
            else:
                print(f", detail={r.text[:200]}")
        except Exception as e:
            print(f"item {iid}: request failed: {type(e).__name__}: {e}")

if __name__ == '__main__':
    check_health()
    test_images()
