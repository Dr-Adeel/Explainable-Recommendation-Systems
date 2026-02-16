import subprocess
import csv
from pathlib import Path
import re
import json
import requests

ROOT = Path('.')
REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)

# configuration
processed_dir = 'data/amazon/processed_small'
als_dir = f'{processed_dir}/als'
ks = [1,5,10,20]
splits = ['valid','test']
max_users = 2000
pool = 2000
relevance_threshold = 4.0
python_exec = Path('.venv') / 'Scripts' / 'python.exe'
if not python_exec.exists():
    python_exec = 'python'

results = []

eval_script = Path('scripts') / 'eval_amazon_als.py'
if not eval_script.exists():
    raise FileNotFoundError('scripts/eval_amazon_als.py not found')

for split in splits:
    for k in ks:
        cmd = [str(python_exec), str(eval_script), '--processed_dir', processed_dir, '--als_dir', als_dir,
               '--split', split, '--k', str(k), '--pool', str(pool), '--relevance_threshold', str(relevance_threshold), '--max_users', str(max_users)]
        print('Running:', ' '.join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = proc.stdout + '\n' + proc.stderr
        # parse evaluated users
        eval_match = re.search(r'Evaluated users:\s*(\d+)', out)
        evaluated = int(eval_match.group(1)) if eval_match else None
        # parse Precision and Recall lines
        prec_match = re.search(r'Precision@%s:\s*([0-9.]+)' % k, out)
        rec_match = re.search(r'Recall@%s:\s*([0-9.]+)' % k, out)
        precision = float(prec_match.group(1)) if prec_match else None
        recall = float(rec_match.group(1)) if rec_match else None
        print(f'-> split={split} k={k} precision={precision} recall={recall} eval_users={evaluated}')
        results.append({'split': split, 'k': k, 'precision': precision, 'recall': recall, 'evaluated_users': evaluated, 'raw_output': out})

# write CSV
csv_path = REPORTS / 'eval_precision_recall.csv'
with csv_path.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['split','k','precision','recall','evaluated_users'])
    w.writeheader()
    for r in results:
        w.writerow({k: r[k] for k in ['split','k','precision','recall','evaluated_users']})
print('Wrote', csv_path)

# save raw outputs
for i,r in enumerate(results):
    p = REPORTS / f'eval_output_{r['split']}_k{r['k']}.txt'
    p.write_text(r['raw_output'], encoding='utf-8')

# --- examples from API ---
api_base = 'http://127.0.0.1:8001'
examples_dir = REPORTS / 'examples'
examples_dir.mkdir(exist_ok=True)

# pick sample item from embeddings preview if exists
emb_preview = REPORTS / 'embeddings_preview.csv'
sample_item = None
if emb_preview.exists():
    import csv
    with emb_preview.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            sample_item = int(rows[0]['item_idx'])

if sample_item is None:
    sample_item = 0

# call similar-items
try:
    r = requests.get(f'{api_base}/amazon/similar-items', params={'item_idx': sample_item, 'k': 10, 'pool': 200, 'filter_category': True}, timeout=10)
    r.raise_for_status()
    out = r.json()
    (examples_dir / f'similar_items_item{sample_item}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('Saved similar-items example')
except Exception as e:
    print('Failed similar-items call:', e)

# call recommend-user for user_idx 0
try:
    r = requests.get(f'{api_base}/amazon/recommend-user', params={'user_idx': 0, 'k': 10, 'pool': 2000}, timeout=10)
    r.raise_for_status()
    out = r.json()
    (examples_dir / 'recommend_user_0.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('Saved recommend-user example')
except Exception as e:
    print('Failed recommend-user call:', e)

# call recommend-hybrid for sample item
try:
    r = requests.get(f'{api_base}/amazon/recommend-hybrid', params={'item_idx': sample_item, 'k': 10, 'pool': 500, 'alpha':0.5, 'beta':0.4, 'gamma':0.1, 'filter_category': True}, timeout=10)
    r.raise_for_status()
    out = r.json()
    (examples_dir / f'recommend_hybrid_item{sample_item}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('Saved recommend-hybrid example')
except Exception as e:
    print('Failed recommend-hybrid call:', e)

print('Done')
