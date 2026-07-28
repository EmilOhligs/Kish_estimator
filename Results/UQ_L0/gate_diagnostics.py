import sys, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT/"src"))
from uq_mace.predictions import load_energies
from uq_mace.reweighting import reweighting_weights

K_B=8.617333262e-5; T=292.0; beta=1.0/(K_B*T); CACHE=ROOT/"cache"
MODELS={"mace-L0-01":("single_mace-L0-01_testbig.npz","firebrick"),
        "mace-L0-c-01":("single_mace-L0-c-01_testbig.npz","darkorange"),
        "ensemble_L2c":("mace_energies_ensemble_L2c_testbig.npz","seagreen")}

def running_maxsum(w,p,rng,reps=30):
    n=w.size; acc=np.zeros(n)
    for _ in range(reps):
        wp=(w[rng.permutation(n)])**p
        acc+=np.maximum.accumulate(wp)/np.cumsum(wp)
    return acc/reps

rng=np.random.default_rng(0)
fig,axes=plt.subplots(2,3,figsize=(15,8))
# Row 1: Max-to-Sum for p=1,2,3 (running, avg over permutations)
for col,p in enumerate([1,2,3]):
    ax=axes[0,col]
    for m,(f,c) in MODELS.items():
        ed,em=load_energies(CACHE/f); w=reweighting_weights(ed,em,beta)
        r=running_maxsum(w,p,rng)
        ax.plot(np.arange(1,w.size+1),r,color=c,lw=1.8,label=m)
    ax.set_title(f"Max-to-Sum  p={p}   " + ("(E[w]?)" if p==1 else "(E[w²]? — Gate)" if p==2 else "(E[w³]?)"),
                 fontweight="bold" if p==2 else "normal")
    ax.set_xlabel("n (gemittelt über 30 Permutationen)"); ax.set_ylabel(f"$R_n(p)=\\max w^{p}/\\sum w^{p}$")
    ax.set_ylim(0,None); ax.grid(alpha=0.3)
    if col==0: ax.legend(fontsize=9)
# Row 2 left: CGF
axC=axes[1,0]
s=np.linspace(-4,1.5,120)
for m,(f,c) in MODELS.items():
    ed,em=load_energies(CACHE/f); dE=ed-em
    K=np.array([np.log(np.mean(np.exp(si*beta*dE))) for si in s])
    axC.plot(s,K,color=c,lw=2,label=m)
axC.axvline(-2,color="k",ls=":",lw=1.5); axC.text(-2.05,axC.get_ylim()[1]*0.75,"E[w²]-Gate (s=-2)",rotation=90,va="top",ha="right",fontsize=9)
axC.axvline(-1,color="grey",ls=":",lw=1); axC.text(-1.05,axC.get_ylim()[1]*0.55,"E[w] (s=-1)",rotation=90,va="top",ha="right",fontsize=8,color="grey")
axC.set_xlabel("s   (t = s·β)"); axC.set_ylabel(r"$\hat K(s\beta)=\log\langle e^{s\beta\Delta E}\rangle$")
axC.set_title("Empirische CGF — glatt & endlich über s=-2 hinaus?"); axC.legend(fontsize=9); axC.grid(alpha=0.3)
# Row 2 mid: R_n(2) final bar vs khat verdict; Row 2 right: left tail of dE (survival of -dE)
axB=axes[1,1]
from uq_mace.reweighting import psis_khat, effective_sample_size
names=[]; R2s=[]; khs=[]; cols=[]
for m,(f,c) in MODELS.items():
    ed,em=load_energies(CACHE/f); w=reweighting_weights(ed,em,beta)
    names.append(m); R2s.append((w**2).max()/(w**2).sum()); khs.append(psis_khat(w)); cols.append(c)
x=np.arange(len(names))
axB.bar(x-0.2,R2s,0.4,color=cols,label="R_N(2)=max/Σ w²",alpha=0.85)
axB.bar(x+0.2,khs,0.4,color=cols,hatch="//",edgecolor="k",label="k̂",alpha=0.5)
axB.axhline(0.5,color="red",ls="--",lw=1.2,label="k̂-Gate 0.5")
axB.set_xticks(x); axB.set_xticklabels(names,rotation=15,fontsize=8); axB.legend(fontsize=8)
axB.set_title("Endwert R_N(2) vs. k̂ (n=400)"); axB.grid(alpha=0.3)
axT=axes[1,2]
for m,(f,c) in MODELS.items():
    ed,em=load_energies(CACHE/f); dE=(ed-em)*1000  # meV
    x_sorted=np.sort(dE)
    # left tail: survival of -dE => small (very negative) dE drives large w
    axT.hist(dE,bins=40,histtype="step",lw=2,color=c,label=m,density=True)
axT.set_xlabel("ΔE [meV]  (linke Flanke = große w)"); axT.set_ylabel("Dichte")
axT.set_title("ΔE-Verteilung — L0 breit/schief, L2c schmal"); axT.legend(fontsize=8); axT.grid(alpha=0.3)
plt.tight_layout()
out=ROOT/"Results"/"UQ_L0"/"gate_diagnostics.png"
plt.savefig(out,dpi=130,bbox_inches="tight"); print("saved",out)
