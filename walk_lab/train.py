"""Cross-entropy direct policy search; each fitness value is a physics rollout."""
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .sim import Simulation,INITIAL,SCALE,LOW,HIGH

_sim=None
def worker(task):
    global _sim
    if _sim is None:_sim=Simulation()
    p,seeds,seconds=task
    results=[_sim.rollout(p,seed=s,seconds=seconds) for s in seeds]
    return float(np.mean([r['score'] for r in results])),[{k:v for k,v in r.items() if k!='frames'} for r in results]

def train(out,generations=50,population=48,workers=4,seed=42,seconds=6,resume=None):
    if generations<1 or population<12 or workers<1 or seed<0 or not 0<seconds<=120:
        raise ValueError('Use generations >= 1, population >= 12, workers >= 1, seed >= 0, and 0 < seconds <= 120.')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed);mean=INITIAL.copy();std=SCALE.copy();history=[];best=mean.copy();best_score=-float('inf');start=time.perf_counter()
    if resume:mean=np.array(json.loads(Path(resume).read_text())['parameters']);best=mean.copy();std=SCALE*.3
    seeds=[11,22]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for generation in range(generations):
            candidates=np.clip(rng.normal(mean,std,(population,len(mean))),LOW,HIGH)
            candidates[0]=best;candidates[1]=mean
            results=list(pool.map(worker,[(p,seeds,seconds) for p in candidates]))
            scores=np.array([r[0] for r in results]);order=np.argsort(scores);elite=candidates[order[-max(4,population//6):]]
            mean=.2*mean+.8*elite.mean(axis=0)
            std=np.maximum(.2*std+.8*elite.std(axis=0),SCALE*.06)
            index=int(order[-1])
            if scores[index]>best_score:best_score=float(scores[index]);best=candidates[index].copy()
            row={'generation':generation+1,'best_score':best_score,'population_median':float(np.median(scores)),
                 'generation_best':float(scores[index]),'elite_duration':float(np.mean([r['duration'] for r in results[index][1]])),
                 'elite_speed':float(np.mean([r['speed'] for r in results[index][1]])),
                 'elapsed_seconds':time.perf_counter()-start}
            history.append(row)
            payload={'algorithm':'CEM','seed':seed,'training_seeds':seeds,'seconds':seconds,'population':population,
              'parameters':best.tolist(),'score':best_score,'history':history,'resume_from':resume}
            (out/'policy.json').write_text(json.dumps(payload,indent=2))
            if generation%5==0 or generation==generations-1:print(json.dumps(row),flush=True)
    return payload

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='runs/training');p.add_argument('--generations',type=int,default=50)
    p.add_argument('--population',type=int,default=48);p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--seconds',type=float,default=6);p.add_argument('--resume');a=p.parse_args()
    train(a.out,a.generations,a.population,a.workers,a.seed,a.seconds,a.resume)
if __name__=='__main__':main()
