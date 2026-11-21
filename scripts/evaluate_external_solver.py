"""Validate native-solver worker output against a specified inventory."""
import json, sys
from pathlib import Path
from flagstone.phase1 import load_json, plot, save_json, validate

source, output = map(Path, sys.argv[1:3])
x = load_json(source)
raw = json.loads(output.read_text())
if raw.get('raw', {}).get('success') is False:
    raise RuntimeError(str(raw['raw']))
seen = set()
for index, px, py, angle in raw['poses']:
    if index in seen or not 0 <= index < len(x.ids): raise ValueError('duplicate or invalid stone ID')
    seen.add(index); x.active[index]=True; x.poses[index]=[px,py,angle]
errors = validate(x,gap=1/24,edge=1/24)
coverage = sum(x.shapes[sid].area for i,sid in enumerate(x.ids) if x.active[i])/x.patio.area
result={'coverage':coverage,'placed':len(seen),'valid':not errors,'violations':errors,
        'elapsed_seconds':raw['elapsed_seconds'],'settings':raw['settings']}
output.with_suffix('.evaluation.json').write_text(json.dumps(result,indent=2))
save_json(x, output.with_suffix('.layout.json'))
plot(x,output.with_suffix('.png'))
print(json.dumps({k:v for k,v in result.items() if k!='violations'},indent=2), 'violations',len(errors))
