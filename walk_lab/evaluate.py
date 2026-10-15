"""Frozen-policy evaluation on seeds not used by the optimizer."""
import argparse,hashlib,json,platform,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import mujoco
from .sim import Simulation,INITIAL

CONDITIONS=[('nominal',.9,0,100),('low_friction',.35,0,200),('forward_push',.9,80,300),('backward_push',.9,-80,400)]

def walking_success(r,seconds):
    # These criteria were set before the held-out evaluation was run.
    return bool(not r['fallen'] and r['duration']>=seconds-.005 and .25<=r['speed']<=1.2
                and r['qualified_alternations']>=8 and min(r['qualified_touchdowns'])>=4)

def summarize(rows):
    return {'trials':len(rows),'successes':sum(r['success'] for r in rows),'falls':sum(r['fallen'] for r in rows),
            'mean_distance_m':float(np.mean([r['distance'] for r in rows])),
            'mean_speed_m_s':float(np.mean([r['speed'] for r in rows])),
            'mean_duration_s':float(np.mean([r['duration'] for r in rows])),
            'mean_alternations':float(np.mean([r['qualified_alternations'] for r in rows])),
            'min_distance_m':float(min(r['distance'] for r in rows)),
            'max_distance_m':float(max(r['distance'] for r in rows))}

def evaluate(policy_file,out,count=20,seconds=12):
    policy_file=Path(policy_file);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    saved=json.loads(policy_file.read_text());parameters=np.array(saved['parameters']);s=Simulation()
    rows=[];replays={}
    for method in ('initial','learned'):
        conditions=CONDITIONS[:1] if method=='initial' else CONDITIONS
        for name,friction,push,base in conditions:
            for seed in range(base,base+count):
                if seed in saved['training_seeds']:raise ValueError('Evaluation seed overlaps training.')
                record=seed==base
                result=s.rollout(INITIAL if method=='initial' else parameters,seed,seconds,friction,push,record)
                frames=result.pop('frames')
                row={**result,'method':method,'condition':name,'seed':seed,'friction':friction,'push_N':push,
                     'success':walking_success(result,seconds)};rows.append(row)
                if record:replays[method+'_'+name]={'label':method+' / '+name,'metrics':row,'frames':frames}
            subset=[r for r in rows if r['method']==method and r['condition']==name]
            print(method,name,json.dumps(summarize(subset)),flush=True)
    summaries={method+'_'+name:summarize([r for r in rows if r['method']==method and r['condition']==name])
               for method,name in dict.fromkeys((r['method'],r['condition']) for r in rows)}
    payload={'created_utc':datetime.now(timezone.utc).isoformat(),'duration_s':seconds,
      'success_definition':'No fall for the full horizon; mean speed 0.25–1.2 m/s; at least 8 alternating landings and 4 landings per foot after >=40 ms airborne.',
      'policy_sha256':hashlib.sha256(policy_file.read_bytes()).hexdigest(),
      'model_sha256':hashlib.sha256(Path(__file__).with_name('humanoid.xml').read_bytes()).hexdigest(),
      'runtime':{'python':sys.version.split()[0],'numpy':np.__version__,'mujoco':mujoco.__version__,'platform':platform.platform()},
      'training':{k:v for k,v in saved.items() if k!='parameters'},'summary':summaries,'trials':rows}
    (out/'evaluation.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    (out/'replays.json').write_text(json.dumps(replays,separators=(',',':')),encoding='utf-8')
    return payload,replays

def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',default='artifacts/policy.json');p.add_argument('--out',default='reports');p.add_argument('--count',type=int,default=20);p.add_argument('--seconds',type=float,default=12)
    a=p.parse_args()
    if not 1<=a.count<=50:p.error('count must be 1–50 to keep seed groups disjoint')
    evaluate(a.policy,a.out,a.count,a.seconds)
if __name__=='__main__':main()
