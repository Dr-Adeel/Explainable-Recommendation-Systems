from importlib import import_module
m = import_module('src.api.amazon_api')
print('module loaded')
print('has compute:', hasattr(m, '_compute_clip_embedding_for_item'))
vec = m._compute_clip_embedding_for_item(4695)
print('vec is', type(vec))
if vec is not None:
    import numpy as np
    print('len', len(vec), 'norm', float(np.linalg.norm(vec)))
else:
    print('compute returned None')
