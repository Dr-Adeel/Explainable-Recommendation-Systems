import requests
items = [2556, 2554, 742]
for item in items:
    try:
        r = requests.get('http://127.0.0.1:8001/amazon/recommend-hybrid', params={'item_idx':item,'k':5,'pool':200,'alpha':0.5,'beta':0.4,'gamma':0.1,'filter_category':True}, timeout=10)
        print(item, r.status_code)
        print(r.json())
    except Exception as e:
        print(item, 'error', e)
