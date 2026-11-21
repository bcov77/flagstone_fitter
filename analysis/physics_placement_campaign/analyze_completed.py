"""Analyze the complete frozen experiment; no simulation or modifications to inputs.
Run with repository .venv (pandas/scipy/matplotlib). Models use pre-trial inputs
and spatially grouped out-of-fold predictions; no independent-layout claim.
"""
import argparse,json,math,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from shapely.geometry import Polygon


def flatten_features(case, context, areas):
    f=case['features'];full=f['scale_profiles']['1.0'];small=f['scale_profiles']['0.9']
    old_area=areas[case['old_id']] if case['family']=='replacement' else 0.
    result={'area_sqft':f['area_sqft'],'potential_area_gain':f['area_sqft']-old_area,
      'old_area_sqft':old_area,'intrusion_sqft':full['intrusion_sqft'],
      'intrusion_fraction':full['intrusion_sqft']/f['area_sqft'],
      'mean_excess_inches':full['mean_excess_inches'],'near_fraction':full['near_fraction'],
      'matching_run_fraction':full['longest_near_run_ft']/f['perimeter_ft'],
      'tangent_alignment':f['near_boundary_tangent_abs_cosine'],
      'convexity':f['convexity'],'wall_distance_inches':f['wall_distance_inches'],
      'neighbor_count':f['neighbor_count_within_half_ft'],'board_stones':context['active_count'],
      'replacement':int(case['family']=='replacement'),'cavity_radius':f['cavity_peak_radius_ft'],
      'width_to_peak':f['minimum_width_to_peak_diameter'],
      'opening_free_area':f['local_clearance_free_area_sqft'],
      'min_width':min(f['widths_15deg_ft']),
      'max_width':max(f['widths_15deg_ft']),
      'scaled_intrusion_fraction':small['intrusion_sqft']/f['area_sqft'],
      'intrusion_growth':(full['intrusion_sqft']-small['intrusion_sqft'])/f['area_sqft'],
      'scaled_excess_inches':small['mean_excess_inches'],
      'scale_excess_change':full['mean_excess_inches']-small['mean_excess_inches'],
      'scale_near_change':full['near_fraction']-small['near_fraction']}
    return result


def build_tables(out):
    m=json.loads((out/'manifest.json').read_text());expected={t['id']:t for t in m['tasks']}
    source=json.loads((out/'source.json').read_text());areas={str(s['id']):Polygon(s['outline']).area for s in source['stones']}
    cases={c['id']:c for c in m['cases']};contexts={c['id']:c for c in m['contexts']};targets={t['id']:t for t in m['targets']}
    rows=[];seen=set()
    for path in sorted((out/'results').glob('*.json')):
        r=json.loads(path.read_text());task=r['task'];tid=task['id'];case=cases[task['case']]
        assert tid in expected and tid==path.stem and r['manifest_hash']==m['manifest_hash'] and r['status']=='complete'
        assert task==expected[tid] and tid not in seen
        seen.add(tid)
        q=r['after_quality'];shifts=r['translation_inches_by_id'];sid=case['candidate']['id'] if case['candidate'] else None
        neighbor_shifts=[v for key,v in shifts.items() if key!=sid]
        row={'id':tid,'case':case['id'],'target':case['target'],'board':case['context'],'family':case['family'],
             'kind':case['kind'],'control':case['control'],'group':targets[case['target']]['region_group'],
             'seed':task['seed'],'protocol':task['protocol'],'valid':r['after']['valid'],
             'converged':r['converged'],'success':bool(r['after']['valid'] and r['converged']),
             'pair_deficit':r['after']['pair_deficit_inches'],'pair_violations':r['after']['pair_violations'],
             'wall_violations':r['after']['wall_violations'],'covered_area':q['covered_area_sqft'],
             'void':q['broad_void_1'],'far_void':q['far_void_1'],
             'void_half':q['broad_void_0.5'],'void_onehalf':q['broad_void_1.5'],
             'quiet_seconds':r['quiet_seconds'],'wall_seconds':r['wall_seconds'],
             'max_neighbor_shift_inches':max(neighbor_shifts,default=0),
             'mean_neighbor_shift_inches':float(np.mean(neighbor_shifts)) if neighbor_shifts else 0,
             'max_speed':r['final_motion']['max_speed_ips']}
        rows.append(row)
    assert seen==set(expected),(len(seen),len(expected))
    df=pd.DataFrame(rows);lookup=df.set_index(['case','seed','protocol'])
    for column in ['covered_area','void','far_void','void_half','void_onehalf']:
        df['delta_'+column]=[row[column]-lookup.loc[(row['control'],row['seed'],row['protocol']),column] for _,row in df.iterrows()]
    df['delta_local_void']=df['delta_void']-df['delta_far_void']
    df['remote_cost']=df['success'] & (df['delta_local_void']<-.02)&(df['delta_far_void']>.02)
    df['useful']=df['success']&(df['delta_covered_area']>=-1e-8)&(df['delta_void']<-.02)&~df['remote_cost']
    df['quality_improves']=df['success']&(df['delta_covered_area']>=-1e-8)&(df['delta_void']<-.02)
    df['label']=np.select([~df['success'],df['remote_cost'],df['useful']],
                         ['invalid_or_unfinished','local_benefit_remote_cost','provisional_quality_improvement'],default='feasible_other')
    is_candidate=~df['kind'].isin(['control','empty_diagnostic']);candidate=df[is_candidate].copy()
    features=[]
    for case in m['cases']:
        if case['kind'] in ['control','empty_diagnostic']:continue
        record={'case':case['id'],'target':case['target'],'family':case['family'],'kind':case['kind'],
                'board':case['context'],'group':targets[case['target']]['region_group']}
        record.update(flatten_features(case,contexts[case['context']],areas));features.append(record)
    feature_df=pd.DataFrame(features)
    averages=candidate.groupby('case').agg(success_rate=('success','mean'),useful_rate=('useful','mean'),
        quality_rate=('quality_improves','mean'),remote_rate=('remote_cost','mean'),
        area_gain=('delta_covered_area','mean'),void_change=('delta_void','mean'),
        far_void_change=('delta_far_void','mean'),max_shift=('max_neighbor_shift_inches','mean'),
        repeats=('id','size')).reset_index()
    case_df=feature_df.merge(averages,on='case',validate='one_to_one')
    assert len(case_df)==960 and (case_df['repeats']==4).all()
    return m,df,candidate,case_df


def weighted_auc(y,p):
    # Case means stand for four matched repeats; no independence assumption/CI.
    order=np.argsort(p);y=np.asarray(y)[order];p=np.asarray(p)[order]
    positive=y;negative=1-y;total=positive.sum()*negative.sum()
    if total==0:return None
    score=0.;prior=0.
    for value in np.unique(p):
        mask=p==value;score+=positive[mask].sum()*(prior+.5*negative[mask].sum());prior+=negative[mask].sum()
    return float(score/total)


def fit_predict(train,test,columns,outcome):
    X=train[columns].astype(float).to_numpy();Z=test[columns].astype(float).to_numpy()
    med=np.nanmedian(X,axis=0);med=np.nan_to_num(med)
    X=np.where(np.isfinite(X),X,med);Z=np.where(np.isfinite(Z),Z,med)
    mean=X.mean(axis=0);std=X.std(axis=0);std[std<1e-8]=1
    X=np.c_[np.ones(len(X)),(X-mean)/std];Z=np.c_[np.ones(len(Z)),(Z-mean)/std]
    y=train[outcome].to_numpy();penalty=1.0
    def objective(beta):
        eta=X@beta;p=expit(eta)
        loss=np.sum(np.logaddexp(0,eta)-y*eta)+.5*penalty*np.dot(beta[1:],beta[1:])
        grad=X.T@(p-y);grad[1:]+=penalty*beta[1:]
        return loss,grad
    result=minimize(objective,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500})
    if not result.success:raise RuntimeError(result.message)
    return expit(Z@result.x)


def model_comparison(cases):
    groups=sorted(cases['group'].unique());np.random.default_rng(20261116).shuffle(groups)
    folds={group:i%5 for i,group in enumerate(groups)}
    cases=cases.copy();cases['fold']=cases['group'].map(folds)
    basic=['area_sqft','potential_area_gain','intrusion_fraction','board_stones','replacement','wall_distance_inches']
    edge=['mean_excess_inches','near_fraction','matching_run_fraction','tangent_alignment','neighbor_count']
    scaled=['scaled_intrusion_fraction','intrusion_growth','scaled_excess_inches','scale_excess_change','scale_near_change']
    opening=['width_to_peak','cavity_radius','opening_free_area','min_width','max_width','convexity']
    feature_sets={'area_intrusion':basic,'plus_boundary':basic+edge,'plus_scale':basic+edge+scaled,
                  'plus_opening':basic+edge+scaled+opening}
    metrics=[]
    for outcome in ['success_rate','useful_rate']:
        y=cases[outcome].to_numpy()
        for name,columns in feature_sets.items():
            p=np.zeros(len(cases))
            for fold in range(5):
                test=cases['fold']==fold
                p[test]=fit_predict(cases[~test],cases[test],columns,outcome)
            p=np.clip(p,1e-6,1-1e-6);cases[f'{outcome}_{name}_prediction']=p
            metrics.append({'outcome':outcome,'features':name,'auc':weighted_auc(y,p),
                            'logloss':float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p))),
                            'case_mean_brier':float(np.mean((p-y)**2))})
    rankings={
        'random':None,'area_only':'potential_area_gain','area_overlap':None,
        **{name:f'useful_rate_{name}_prediction' for name in feature_sets}}
    rank_rows=[]
    for target,subset in cases.groupby('target'):
        for name,column in rankings.items():
            if name=='random':selected=subset
            elif name=='area_overlap':selected=subset.assign(score=subset['area_sqft']-8*subset['intrusion_sqft']).nlargest(3,'score')
            else:selected=subset.nlargest(3,column)
            rank_rows.append({'target':target,'group':subset['group'].iloc[0],'family':subset['family'].iloc[0],
                             'ranker':name,'top3_useful_rate':float(selected['useful_rate'].mean()),
                             'top3_success_rate':float(selected['success_rate'].mean())})
    return cases,pd.DataFrame(metrics),pd.DataFrame(rank_rows)


def analyze(out):
    result_dir=out/'completed_analysis';result_dir.mkdir(exist_ok=True)
    m,df,candidates,cases=build_tables(out)
    df.to_csv(result_dir/'trial_outcomes.csv',index=False)
    cases,metrics,rankings=model_comparison(cases)
    cases.to_csv(result_dir/'case_features_outcomes.csv',index=False)
    metrics.to_csv(result_dir/'grouped_model_comparison.csv',index=False)
    rankings.to_csv(result_dir/'target_ranking_comparison.csv',index=False)
    by_kind=candidates.groupby(['family','kind']).agg(trials=('id','size'),success_rate=('success','mean'),
        useful_rate=('useful','mean'),remote_cost_rate=('remote_cost','mean'),mean_area_gain=('delta_covered_area','mean'),
        mean_void_change=('delta_void','mean'),mean_max_shift=('max_neighbor_shift_inches','mean')).reset_index()
    by_kind.to_csv(result_dir/'proposal_kind_comparison.csv',index=False)
    paired=candidates.pivot(index=['case','seed'],columns='protocol',values=['success','useful','delta_void','delta_far_void'])
    direct=paired[('success','direct')].astype(bool);cycle=paired[('success','gap_cycle')].astype(bool)
    dp=paired[('useful','direct')].astype(bool);cp=paired[('useful','gap_cycle')].astype(bool)
    pair_summary={'paired_case_seed_count':len(paired),'both_feasible':int((direct&cycle).sum()),
        'only_direct_feasible':int((direct&~cycle).sum()),'only_cycle_feasible':int((~direct&cycle).sum()),
        'neither_feasible':int((~direct&~cycle).sum()),'only_direct_useful':int((dp&~cp).sum()),
        'only_cycle_useful':int((~dp&cp).sum()),'both_useful':int((dp&cp).sum()),
        'mean_cycle_minus_direct_void':float((paired[('delta_void','gap_cycle')]-paired[('delta_void','direct')]).mean())}
    paired.to_csv(result_dir/'paired_protocols.csv')
    # Aggregate thresholds without pretending the threshold has been validated.
    bins=pd.cut(cases['intrusion_fraction'],[-1e-8,.05,.25,.5,1.000001],labels=['0–5%','5–25%','25–50%','50–100%'])
    intrusion=cases.assign(intrusion_bin=bins).groupby('intrusion_bin',observed=True).agg(cases=('case','size'),
       success_rate=('success_rate','mean'),useful_rate=('useful_rate','mean'),mean_max_shift=('max_shift','mean')).reset_index()
    intrusion.to_csv(result_dir/'intrusion_bands.csv',index=False)
    sensitivity=[]
    for threshold in [.01,.02,.05]:
        remote=candidates['success']&(candidates['delta_local_void']<-threshold)&(candidates['delta_far_void']>threshold)
        useful=candidates['success']&(candidates['delta_covered_area']>=-1e-8)&(candidates['delta_void']<-threshold)&~remote
        sensitivity.append({'threshold_sqft':threshold,'useful':int(useful.sum()),'remote_cost':int(remote.sum())})
    summary={'planned':len(m['tasks']),'completed':len(df),'candidate_trials':len(candidates),'candidate_piece_poses':len(cases),
       'targets':len(m['targets']),'contexts':len(m['contexts']),'spatial_groups':cases['group'].nunique(),
       'valid_all':int(df['valid'].sum()),'converged_all':int(df['converged'].sum()),
       'controls':int((df['kind']=='control').sum()),'controls_valid':int(df.loc[df['kind']=='control','valid'].sum()),
       'controls_success':int(df.loc[df['kind']=='control','success'].sum()),
       'candidate_valid':int(candidates['valid'].sum()),'candidate_success':int(candidates['success'].sum()),
       'candidate_labels':dict(Counter(candidates['label'])),'paired_protocols':pair_summary,
       'threshold_sensitivity':sensitivity,
       'protocol_rates':candidates.groupby('protocol')[['success','useful','remote_cost']].mean().to_dict('index'),
       'consistent_feasible_cases':int((cases['success_rate']==1).sum()),'never_feasible_cases':int((cases['success_rate']==0).sum()),
       'mixed_feasibility_cases':int(((cases['success_rate']>0)&(cases['success_rate']<1)).sum()),
       'consistent_useful_cases':int((cases['useful_rate']==1).sum()),
       'max_neighbor_shift_inches':float(candidates['max_neighbor_shift_inches'].max()),
       'neighbor_shifts_above_6_inches':int((candidates['max_neighbor_shift_inches']>6).sum()),
       'same_source_lineages':1,'generalization':'Grouped within-layout diagnostic; not independent-layout validation'}
    (result_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print('\nMETHODS\n',by_kind.to_string(index=False))
    print('\nMODELS\n',metrics.to_string(index=False))
    print('\nTOP3\n',rankings.groupby('ranker')[['top3_useful_rate','top3_success_rate']].mean().to_string())
    print('\nINTRUSION\n',intrusion.to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path(__file__).with_name('campaign_v1'))
    analyze(parser.parse_args().out)
