import sys
import json

with open('/app/isp_analysis.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells'][-5:]):
    print(f"--- Cell {len(nb['cells'])-5+i} [{cell['cell_type']}] ---")
    source = cell.get('source', [])
    if isinstance(source, list):
        source = "".join(source)
    print(source[:1000])
