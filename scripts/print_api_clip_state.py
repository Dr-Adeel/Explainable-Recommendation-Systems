from importlib import import_module
m = import_module('src.api.amazon_api')
print('module loaded')
print('_X is None?', m._X is None)
print('_item_ids len', len(m._item_ids) if m._item_ids is not None else None)
print('_X shape', None if m._X is None else m._X.shape)
print('first 5 item_ids', None if m._item_ids is None else m._item_ids[:5])
print('clip_ready check in module health:', getattr(m, 'health')())
