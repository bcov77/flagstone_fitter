"""Single-thread sequential native portfolio; no uncontrolled anytime thread pool."""
import argparse
import json
import random
import time
from pathlib import Path
import quarter_native


def run(input_file,output,seconds):
    data=json.loads(Path(input_file).read_text());out=Path(output);out.mkdir(exist_ok=False)
    started=time.monotonic();rng=random.Random(data.get('seed',0));patterns=[];iteration=0
    while time.monotonic()-started<seconds:
        candidate=dict(data,items=list(data['items']))
        candidate.pop('optimization_mode',None);rng.shuffle(candidate['items'])
        path=out/f'input_{iteration:04d}.json';path.write_text(json.dumps(candidate))
        directory=out/f'round_{iteration:04d}'
        quarter_native.run(path,directory,max(.05,seconds-(time.monotonic()-started)))
        payload=json.loads((directory/'patterns.json').read_text()) if (directory/'patterns.json').exists() else {}
        patterns.extend(payload.get('patterns',[]))
        # Keep a first construction and recent alternatives; original outlines
        # and new objective select from these in the scientific worker.
        if len(patterns)>24:patterns=[patterns[0],*patterns[-23:]]
        tmp=out/'patterns.tmp';tmp.write_text(json.dumps(dict(patterns=patterns,native_workers=1,
                  mode='sequential input-order portfolio',iterations=iteration+1,acceptance_authority=False)))
        tmp.replace(out/'patterns.json');iteration+=1
        if not payload.get('patterns'):break


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--seconds',type=float,required=True)
    a=p.parse_args();run(a.input,a.output,a.seconds)
