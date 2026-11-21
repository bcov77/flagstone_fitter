"""Pre-trial original-piece descriptors missing from the original proposal table.
Computes geometry only, never simulates or modifies frozen inputs.
"""
import json
import campaign as c

def main():
 out=c.DEFAULT;m=c.load_manifest(out);source=json.loads((out/'source.json').read_text())
 contexts={x['id']:x for x in m['contexts']};cases={x['id']:x for x in m['cases']};result={}
 for target in m['targets']:
  if target['family']!='replacement':continue
  control=cases[target['id']+'_control'];document=c.make_document(source,contexts[target['context']],control)
  result[target['id']]=c.features(document,control,target['radius_ft'])
 (out/'completed_analysis/old_piece_features.json').write_text(json.dumps(result,indent=2))
 print('Measured original-piece geometry for',len(result),'replacement targets')
if __name__=='__main__':main()
