"""Tune PID gains for SutureController with a minimal CMA-ES.
Writes artifacts/best_gains.json and artifacts/tuning_history.json.
Usage: python tune_gains.py --seeds 6 --generations 12 --popsize 12
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import mujoco

from run_micro_suture_lab import (
    SCENE, ARTIFACTS, Gains, ModelIndex, rollout, GAINS_LOWER, GAINS_UPPER,
)

def make():
    m=mujoco.MjModel.from_xml_path(str(SCENE)); d=mujoco.MjData(m); i=ModelIndex(m)
    return m,d,i

def objective(vec, model, data, idx, seeds, duration, dt, noise_scale):
    v=np.clip(vec,GAINS_LOWER,GAINS_UPPER)
    g=Gains.from_vector(v); scores=[]
    for s in seeds:
        rng=np.random.default_rng(20260619+int(s))
        final,_=rollout(model,data,idx,g,duration,dt,use_feedback=True,rng=rng,noise_scale=noise_scale)
        scores.append(final["task_score"])
    return -float(np.mean(scores)), float(np.std(scores))

# ----- minimal (mu,lambda)-CMA-ES -----
def cma_es(f, x0, sigma0, popsize, generations, log=None):
    n=len(x0); mu=popsize//2
    weights=np.log(mu+0.5)-np.log(1+np.arange(mu)); weights/=weights.sum()
    mu_eff=1.0/np.sum(weights**2)
    cs=(mu_eff+2)/(n+mu_eff+5); cc=(4+mu_eff/n)/(n+4+2*mu_eff/n)
    c1=2/((n+1.3)**2+mu_eff); cmu=min(1-c1, 2*(mu_eff-2+1/mu_eff)/((n+2)**2+mu_eff))
    damps=1+2*max(0,np.sqrt((mu_eff-1)/(n+1))-1)+cs
    mean=np.array(x0,dtype=float); C=np.eye(n); ps=np.zeros(n); pc=np.zeros(n); sigma=float(sigma0)
    chiN=np.sqrt(n)*(1-1/(4*n)+1/(21*n**2))
    best=(np.inf,mean.copy())
    history=[]
    for gen in range(generations):
        D2,B=np.linalg.eigh(C); D=np.sqrt(np.maximum(D2,1e-20))
        samples=[mean+sigma*(B@(D*np.random.randn(n))) for _ in range(popsize)]
        fits=[f(s) for s in samples]
        fitness=np.array([fv[0] for fv in fits])
        order=np.argsort(fitness)
        S=np.array([samples[i] for i in order[:mu]])
        old_mean=mean.copy()
        mean=weights@S
        if fitness[order[0]]<best[0]:
            best=(float(fitness[order[0]]),samples[order[0]].copy())
        C_inv_sqrt=B@np.diag(1.0/D)@B.T
        ps=(1-cs)*ps + np.sqrt(cs*(2-cs)*mu_eff)*(C_inv_sqrt@((mean-old_mean)/sigma))
        hs=np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(gen+1)))/chiN<(1.4+2/(n+1))
        pc=(1-cc)*pc + (1.0 if hs else 0.0)*np.sqrt(cc*(2-cc)*mu_eff)*((mean-old_mean)/sigma)
        artmp=(S-old_mean)/sigma
        C=((1-c1-cmu)*C + c1*(np.outer(pc,pc) + (0.0 if hs else cc*(2-cc))*C)
            + cmu*(artmp.T@np.diag(weights)@artmp))
        sigma=sigma*np.exp((cs/damps)*(np.linalg.norm(ps)/chiN-1))
        gen_log={"gen":gen,"sigma":float(sigma),"best_neg_score":best[0],
                 "mean_neg_score":float(fitness.mean()),
                 "best_so_far_vec":best[1].tolist(),"mean_vec":mean.tolist()}
        history.append(gen_log)
        if log: log(gen_log)
    return best[1],best[0],history

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,default=6)
    p.add_argument("--generations",type=int,default=12)
    p.add_argument("--popsize",type=int,default=12)
    p.add_argument("--duration",type=float,default=4.0)
    p.add_argument("--dt",type=float,default=0.001)
    p.add_argument("--sigma",type=float,default=0.25)
    p.add_argument("--noise",type=float,default=1.0)
    a=p.parse_args()
    m,d,i=make()
    seeds=list(range(a.seeds))
    x0=Gains().to_vector()
    def f(v): return objective(v,m,d,i,seeds,a.duration,a.dt,a.noise)
    t0=time.time()
    best_vec,best_neg,hist=cma_es(f,x0,a.sigma,a.popsize,a.generations,
                                  log=lambda g:print(f"[gen {g['gen']:02d}] best={-g['best_neg_score']:.4f} sigma={g['sigma']:.3f}"))
    elapsed=time.time()-t0
    best_vec=np.clip(best_vec,GAINS_LOWER,GAINS_UPPER)
    final_score=-best_neg
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    (ARTIFACTS/"best_gains.json").write_text(json.dumps({
        "gains_vector":best_vec.tolist(),"gains":Gains.from_vector(best_vec).__dict__,
        "score_mean":final_score,"elapsed_s":round(elapsed,2),
        "seeds":a.seeds,"generations":a.generations,"popsize":a.popsize,
    },indent=2))
    (ARTIFACTS/"tuning_history.json").write_text(json.dumps(hist,indent=2))
    print(json.dumps({"best_gains":Gains.from_vector(best_vec).__dict__,
                      "score":final_score,"elapsed_s":round(elapsed,2)},indent=2))

if __name__=="__main__": main()
