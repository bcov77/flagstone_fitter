"""Small physics check of the retrieved bar, plus recorded-pose controls."""
import copy
import gzip
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from pilot import run_physics,evaluate,canonical
from study import load,document
from physics import relax


def main():
    source,*_=load();case=next(c for c in json.loads((HERE.parent/'human_trajectory_fa791034/fixtures.json').read_text()) if c['id']=='run_21')
    with gzip.open(HERE/'run_21/result.json.gz','rt') as f:pilot=json.load(f)
    auto=next(t['action'] for t in pilot['trials'] if t['action']['family']=='mobile_seam')
    inputs=[('autonomous',pilot['nodes'][0]['document'],auto),
            ('recorded_control',document(source,case['pre']),dict(remove=[],placements=[dict(stone='13',pose=case['proposed']['pieces']['13']['pose'])]))]
    cfg=json.loads((HERE/'config.json').read_text());protocols=['short_pulses','long_quiet','held_then_release'];dest=HERE/'seam_trials';dest.mkdir(exist_ok=True)
    manifest=dict(script_sha256=canonical(Path(__file__).read_text()),pilot_sha256=canonical(pilot),
                  inputs=[dict(name=n,document=d,action=a) for n,d,a in inputs],config=cfg,protocols=protocols,seed=1701,
                  scope='Two starts x three protocols, one seed. Recorded control is diagnostic and is not autonomous.')
    if (dest/'manifest.json').exists():assert json.loads((dest/'manifest.json').read_text())==manifest
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name,doc,action in inputs:
        for protocol in protocols:
            path=dest/f'{name}_{protocol}.json.gz'
            if path.exists():continue
            config=copy.deepcopy(cfg)
            if protocol=='held_then_release':
                config=json.loads((HERE.parent/'physics_trajectories/config.json').read_text());config['quiet_max_seconds']=12
                result=relax(doc,action,config,1701);end=result.pop('attempted_document');result['metrics']=evaluate(end)
            else:
                config['quiet_max_seconds']=12 if protocol=='short_pulses' else 20
                if protocol=='short_pulses':config['pulse_seconds']=case['pulse_durations']
                end,result=run_physics(doc,action,config,1701)
            result.update(document=end,action=action,protocol=protocol,input=name,config=config,seed=1701)
            with gzip.open(path,'wt') as f:json.dump(result,f)
            print(name,protocol,'accepted',result['accepted'],'coverage',result['metrics']['coverage'],
                  'pairs/walls',result['metrics']['pair_failures'],result['metrics']['wall_failures'],flush=True)


if __name__=='__main__':main()
