from pathlib import Path
import math
import mujoco
import numpy as np

DT=.01
# Shared leg waveform: bias + sine + cosine + second harmonic.
# The second leg is half a cycle behind; feedback adjusts both hips/ankles.
INITIAL=np.array([1.1, -.08,.24,-.12, .22,.28,-.12, 0.,.12,0., 0.,0.,0., 1.,.15,.08])
SCALE=np.array([.35, .18,.18,.18, .25,.25,.20, .20,.20,.20, .12,.12,.12, .5,.15,.15])
LOW=np.array([.5,-.5,.02,-.5,-.7,-.7,-.6,-.7,-.7,-.6,-.4,-.4,-.4,0,0,-.5])
HIGH=np.array([2.,.5,.8,.5,.7,.7,.6,.7,.7,.6,.4,.4,.4,3.,1.,.5])

class Simulation:
    def __init__(self):
        self.model=mujoco.MjModel.from_xml_path(str(Path(__file__).with_name('humanoid.xml')))
        self.data=mujoco.MjData(self.model)
        self.feet=[self.model.geom(x).id for x in ('left_sole','right_sole')]
    def reset(self,seed=0,friction=.9,push=0):
        mujoco.mj_resetData(self.model,self.data)
        rng=np.random.default_rng(seed)
        self.data.qpos[2]=rng.uniform(-.015,.015)
        self.data.qvel[:]=rng.normal(0,.01,self.model.nv)
        self.model.geom_friction[:,0]=friction
        self.push=push
        mujoco.mj_forward(self.model,self.data)
    def action(self,p):
        d=self.data
        phase=2*math.pi*p[0]*d.time
        targets=[]
        for offset in (0,math.pi):
            t=phase+offset
            target=p[1:4]+p[4:7]*math.sin(t)+p[7:10]*math.cos(t)+p[10:13]*math.sin(2*t)
            target=target.copy()
            target[0]+=p[13]*d.qpos[2]+p[14]*d.qvel[2]
            target[2]-=d.qpos[2]+p[15]*(d.qvel[0]-.6)
            targets.extend(target)
        return np.clip(np.tile([110,130,65],2)*(targets-d.qpos[3:])-np.tile([5,5,3],2)*d.qvel[3:],[-80,-80,-35]*2,[80,80,35]*2)
    def rollout(self,p,seed=0,seconds=6,friction=.9,push=0,record=False):
        p=np.asarray(p,dtype=float)
        if p.shape!=INITIAL.shape or not np.isfinite(p).all():raise ValueError('Expected 16 finite policy parameters.')
        if not 0<seconds<=120 or not 0<friction<=3 or not np.isfinite(push) or abs(push)>200:
            raise ValueError('Use 0 < seconds <= 120, 0 < friction <= 3, and a finite push within +/-200 N.')
        self.reset(seed,friction,push);d=self.data;m=self.model
        reward=energy=0.;frames=[];contacts=np.zeros(2,dtype=bool);touchdowns=[0,0];last_touch=-1;alternations=0
        air_steps=np.zeros(2,dtype=int);qualified=[0,0];qualified_alternations=0;last_qualified=-1;peak_clearance=[0.,0.]
        start=float(d.qpos[0]);steps=round(seconds/DT);fallen=False
        for i in range(steps):
            d.ctrl[:]=self.action(p)
            d.xfrc_applied[1,0]=push if 2.0<=d.time<2.1 else 0
            mujoco.mj_step(m,d,nstep=5)
            v=float(d.qvel[0]);tilt=float(d.qpos[2]);height=float(d.qpos[1])
            power=float(np.sum(np.abs(d.ctrl*d.qvel[3:])));energy+=power*DT
            reward+=(.3+1.5*math.exp(-((v-.6)/.45)**2)-.8*tilt**2-.0004*power)*DT
            current=np.array([any(g in (c.geom1,c.geom2) for c in d.contact[:d.ncon]) for g in self.feet])
            for foot in range(2):
                if current[foot] and not contacts[foot] and i>10:
                    touchdowns[foot]+=1
                    if last_touch!=-1 and last_touch!=foot:alternations+=1
                    last_touch=foot
                if current[foot] and air_steps[foot]>=4 and i>10:
                    qualified[foot]+=1
                    if last_qualified!=-1 and last_qualified!=foot:qualified_alternations+=1
                    last_qualified=foot
                air_steps[foot]=0 if current[foot] else air_steps[foot]+1
                g=self.feet[foot]
                clearance=float(d.geom_xpos[g,2]-np.dot(np.abs(d.geom_xmat[g].reshape(3,3)[2]),m.geom_size[g]))
                peak_clearance[foot]=max(peak_clearance[foot],clearance)
            contacts=current
            if record and i%3==0:
                frames.append({'t':round(float(d.time),4),'qpos':d.qpos.tolist(),'bodies':d.xpos.tolist(),'contacts':current.tolist(),'velocity':v,'height':height})
            fallen=not np.isfinite(d.qpos).all() or height<.48 or abs(tilt)>1.0
            if fallen:break
        elapsed=float(d.time);distance=float(d.qpos[0])-start
        return {'score':reward,'duration':elapsed,'distance':distance,'speed':distance/elapsed,'fallen':fallen,'energy':energy,
                'touchdowns':touchdowns,'alternations':alternations,'qualified_touchdowns':qualified,
                'qualified_alternations':qualified_alternations,'peak_foot_clearance':peak_clearance,'frames':frames}
