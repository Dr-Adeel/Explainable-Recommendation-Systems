import requests, json
base='http://127.0.0.1:8001'
items = requests.get(base+'/amazon/sample-items', params={'n':10,'with_images':True}).json().get('items',[])
print('sample count', len(items))
if items:
    it = items[0]['item_idx']
    print('item', it)
    r = requests.get(base+'/amazon/explain-recommendation', params={'item_idx':it,'k':5,'pool':500,'alpha':0.5,'beta':0.4,'gamma':0.1})
    print(json.dumps(r.json(), indent=2))
else:
    print('no items')
