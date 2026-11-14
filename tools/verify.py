import subprocess,numpy as np,glob,os,json
res={}
for f in sorted(glob.glob('release/*.m4a')+glob.glob('bundled/*.m4a')):
    raw=subprocess.run(['ffmpeg','-loglevel','error','-i',f,'-f','f32le','-ac','2','-ar','44100','-'],capture_output=True).stdout
    y=np.frombuffer(raw,np.float32).reshape(-1,2)
    pk=np.max(np.abs(y)); clip=int(np.sum(np.abs(y)>=0.999))
    w=4410; n=len(y)//w; r=20*np.log10(np.sqrt((y[:n*w]**2).reshape(n,w,2).mean((1,2)))+1e-12)
    res[os.path.basename(f)]=dict(dur=round(len(y)/44100,3),peak_db=round(float(20*np.log10(pk)),2),clipped=clip,min_rms_db=round(float(r.min()),1),med_rms_db=round(float(np.median(r)),1),seam=round(float(abs(r[-1]-r[0])),1))
    print(os.path.basename(f),res[os.path.basename(f)])
json.dump(res,open('_work/verify_decoded.json','w'),indent=1)
