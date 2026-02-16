import requests, json
url = 'http://127.0.0.1:8001/amazon/explain-recommendation'
params = {'item_idx': 4695, 'k': 5, 'pool': 500, 'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0}
try:
    r = requests.get(url, params=params, timeout=10)
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print('request failed', e)
