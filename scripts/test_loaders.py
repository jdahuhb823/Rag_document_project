from rag.loader import load_and_split
from pathlib import Path

samples = [Path('sample_data/business_report.txt')]

for s in samples:
    print('\n===', s, '===')
    try:
        docs = load_and_split([str(s)])
        print('Documents returned:', len(docs))
        for i, d in enumerate(docs[:5], 1):
            text = getattr(d, 'page_content', None) or getattr(d, 'text', None) or str(d)
            meta = getattr(d, 'metadata', {})
            print(f'-- Doc {i} len={len(text)} src={meta.get("source")}')
            print(text[:300])
    except Exception as e:
        print('Loader error:', e)
