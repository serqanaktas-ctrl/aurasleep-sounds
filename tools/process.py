import numpy as np, soundfile as sf, pyloudnorm as pyln, json, subprocess, os, hashlib, sys
from catalog import C
from gen import filt, hp, lp
SR=44100; X=int(3*SR); TARGET=-18.0
meter=pyln.Meter(SR)
def load(path):
    x,sr=sf.read(path,dtype='float64',always_2d=True); assert sr==SR,(path,sr)
    if x.shape[1]==1: x=np.repeat(x,2,1)
    return x[:,:2]
def xfade_loop(x, L):
    """seamless loop of length L from x (needs L+X samples): blend tail x[L:L+X] into head."""
    assert len(x)>=L+X,(len(x),L)
    t=np.linspace(0,np.pi/2,X)[:,None]
    y=x[:L].copy(); y[:X]=x[:X]*np.sin(t)+x[L:L+X]*np.cos(t); return y
def rec_loop(key, Lsec=120):
    x=load(f'raw/{key}.wav'); x=x-x.mean(0)
    x=np.stack([filt(x[:,0],hp(30)),filt(x[:,1],hp(30))],1)
    avail=len(x)-X-SR//2
    L=min(Lsec*SR, avail)
    y=xfade_loop(x,L)
    if L<60*SR: y=np.tile(y,(int(np.ceil(60*SR/L)),1))
    return y
def gen_loop(key, Lsec=120):
    x=load(f'gen/{key}.wav')
    return x if Lsec*SR>=len(x) else xfade_loop(x,Lsec*SR)
def loud(y): return meter.integrated_loudness(y)
from scipy.ndimage import minimum_filter1d, uniform_filter1d
CEIL=0.708; MAXGR=10**(-12/20)
def limit(y, w=int(0.04*SR)):
    a=np.max(np.abs(y),1); g=np.minimum(1,CEIL/np.maximum(a,1e-9)); g=np.maximum(g,MAXGR)
    g=minimum_filter1d(g,2*w+1,mode='wrap'); g=uniform_filter1d(g,w,mode='wrap')
    return y*g[:,None]
TOVR={'clock':-21.0,'fireplace':-21.0}
def normalize(y,tgt=None):
    TARGET=tgt or globals()['TARGET']; note=''
    for i in range(4):
        y=y*10**((TARGET-loud(y))/20)
        if np.max(np.abs(y))<=CEIL: break
        y=limit(y); note='limited'
    pk=np.max(np.abs(y))
    if pk>CEIL: y=y*CEIL/pk; note=f'limited, {loud(y):.1f} LUFS'
    return y,note
def checks(y):
    pk=np.max(np.abs(y)); w=int(0.1*SR); n=len(y)//w
    rms=np.sqrt((y[:n*w]**2).reshape(n,w,2).mean((1,2)))+1e-12
    rdb=20*np.log10(rms); med=np.median(rdb)
    gaps=int(np.sum(rdb<med-40))
    s=int(0.05*SR); a=np.sqrt(np.mean(y[-s:]**2)); b=np.sqrt(np.mean(y[:s]**2)); seam=abs(20*np.log10((a+1e-12)/(b+1e-12)))
    jump=np.max(np.abs(y[0]-y[-1])); typ=np.percentile(np.abs(np.diff(y,axis=0)),99.9)
    return dict(peak_db=round(20*np.log10(pk),2),silence_windows=gaps,seam_rms_db=round(seam,2),seam_jump=round(float(jump),4),diff_p999=round(float(typ),4),lra_proxy=round(float(np.percentile(rdb,95)-np.percentile(rdb,10)),1))
def encode(y, out, br='96k'):
    tmp=out+'.tmp.wav'; sf.write(tmp,y.astype(np.float32),SR,subtype='FLOAT')
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',tmp,'-c:a','aac','-b:a',br,'-ar','44100','-movflags','+faststart','-map_metadata','-1','-fflags','+bitexact','-flags:a','+bitexact',out],check=True); os.remove(tmp)
os.makedirs('../release',exist_ok=True); os.makedirs('../bundled',exist_ok=True)
only=set(sys.argv[1:]); rep={}
if os.path.exists('qc.json'): rep=json.load(open('qc.json'))
for (id,cat,pro,bun,kind,srcs,tr,en) in C:
    if only and id not in only: continue
    Ls=60 if bun else 120
    if kind=='rec': y=rec_loop(srcs[0],Ls)
    elif kind=='gen': y=gen_loop(srcs[0],Ls)
    else:
        y=None
        for s in srcs:
            t,k,g=s.split(':'); c=rec_loop(k,120) if t=='r' else gen_loop(k,120)
            assert len(c)==120*SR,(k,len(c)/SR)
            c,_=normalize(c); y=c*float(g) if y is None else y+c*float(g)
    y,note=normalize(y,TOVR.get(id))
    out=f"../{'bundled' if bun else 'release'}/{id}.m4a"; encode(y,out)
    q=checks(y); q.update(lufs=round(loud(y),1),dur=round(len(y)/SR,2),note=note,bytes=os.path.getsize(out))
    rep[id]=q; print(id,q,flush=True)
json.dump(rep,open('qc.json','w'),indent=1)
