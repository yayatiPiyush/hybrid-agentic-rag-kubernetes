"""Check provisional evaluation page IDs against your actual indexes/meta.json."""
import json
from collections import Counter
from pathlib import Path

DATASET = Path('kubernetes_eval_40_provisional.json')
METADATA = Path('indexes/meta.json')

if not DATASET.exists() or not METADATA.exists():
    raise SystemExit('Run from project root and ensure dataset and indexes/meta.json exist.')

questions = json.loads(DATASET.read_text(encoding='utf-8'))
meta = json.loads(METADATA.read_text(encoding='utf-8'))
chunks = meta.values() if isinstance(meta, dict) else meta
counts = Counter(c['page_id'] for c in chunks)

missing = []
for item in questions:
    found = [p for p in item['relevant_pages'] if p in counts]
    absent = [p for p in item['relevant_pages'] if p not in counts]
    print(f"{item['id']}: {len(found)}/{len(item['relevant_pages'])} labeled pages indexed - {item['question']}")
    for page in absent:
        print(f'   NOT INDEXED: {page}')
        missing.append({'id':item['id'],'question':item['question'],'page_id':page})

report = {'questions':len(questions), 'missing_labels':missing,
          'label_note':'Existence check only; human relevance judgments are still required.'}
Path('eval_label_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(f'\nMissing labeled page IDs: {len(missing)}. Saved eval_label_validation.json')
print('Note: indexed does NOT mean a page is the only relevant page for that question.')
