import numpy as np, soundfile as sf, sys
SR=44100
rng=np.random.default_rng(20261114)
def nfft(n): return np.fft.rfftfreq(n,1/SR)
def shaped(n, shape, seed=None):
    r=np.random.default_rng(seed) if seed is not None else rng
    X=np.fft.rfft(r.standard_normal(n)); f=nfft(n); f[0]=1
    y=np.fft.irfft(X*shape(f),n); return y/np.std(y)
def filt(x, shape):  # circular (periodic) frequency-domain filter
    f=nfft(len(x)); f[0]=1; return np.fft.irfft(np.fft.rfft(x)*shape(f),len(x))
def lp(fc,o=2): return lambda f:1/np.sqrt(1+(f/fc)**(2*o))
def hp(fc,o=2): return lambda f:1/np.sqrt(1+(fc/f)**(2*o))
def bp(lo,hi,o=2): return lambda f:hp(lo,o)(f)*lp(hi,o)(f)
def mul(*s): return lambda f: np.prod([g(f) for g in s],axis=0)
def lfo(n, cycles, phase=0.0):  # periodic slow modulation, integer cycles over loop
    t=np.arange(n)/n; return np.sin(2*np.pi*cycles*t+phase)
def smooth_rand(n, k=12, seed=0):  # periodic smooth random curve (0..1)
    r=np.random.default_rng(seed); y=sum(r.uniform(.3,1)/c*lfo(n,c,r.uniform(0,6.28)) for c in range(1,k))
    y=(y-y.min())/(y.max()-y.min()); return y
def place(buf, ev, pos):  # wrap-around add
    n=len(buf); idx=(pos+np.arange(len(ev)))%n; np.add.at(buf, idx, ev)
def st(l,r): return np.stack([l,r],1)
def norm(x): return x/np.max(np.abs(x))*0.5
def reverb_circ(x, t60=2.5, mix=0.3, seed=1):
    n=len(x); m=int(SR*t60); r=np.random.default_rng(seed)
    ir=r.standard_normal(m)*np.exp(-6.9*np.arange(m)/m); ir=filt(np.pad(ir,(0,0)),lp(5000))
    irp=np.zeros(n); irp[:m]=ir/np.sqrt(np.sum(ir**2))
    wet=np.fft.irfft(np.fft.rfft(x)*np.fft.rfft(irp),n)
    return (1-mix)*x+mix*wet*np.std(x)/max(np.std(wet),1e-9)

D=120; N=SR*D
def white(): return st(shaped(N,lambda f:np.ones_like(f)),shaped(N,lambda f:np.ones_like(f)))*0.2
def pink(): s=lambda f:1/np.sqrt(f)*hp(20)(f); return st(shaped(N,s),shaped(N,s))*0.2
def brown(): s=lambda f:1/f*hp(25)(f); return st(shaped(N,s),shaped(N,s))*0.2
def deep_brown(): s=lambda f:1/f*hp(20)(f)*lp(180,3)(f); return st(shaped(N,s),shaped(N,s))*0.2
def grey():  # approx inverse equal-loudness: boost lows & highs over pink-ish
    def s(f): return (1/np.sqrt(f))*(1+ (120/f)**1.2*0.0+0)*(1+2.5*np.exp(-((np.log10(f)-1.9)**2)/0.15)+1.2*np.exp(-((np.log10(f)-4.0)**2)/0.08))/(1+0.8*np.exp(-((np.log10(f)-3.5)**2)/0.1))*hp(20)(f)
    return st(shaped(N,s),shaped(N,s))*0.2
def green(): s=lambda f:np.exp(-((np.log2(f/500))**2)/2.2)*hp(30)(f); return st(shaped(N,s),shaped(N,s))*0.2
def fan():
    s=lambda f:1/np.sqrt(f)*lp(900,2)(f)*hp(40)(f)
    l,r=shaped(N,s),shaped(N,s); t=np.arange(N)/SR
    blade=1+0.06*np.sin(2*np.pi*(N/SR*23.0/D)*0+2*np.pi*23.0*t)  # 23 Hz blade pass (integer cycles: 23*120)
    hum=0.05*(np.sin(2*np.pi*100*t)+0.5*np.sin(2*np.pi*200*t)+0.25*np.sin(2*np.pi*300*t))
    return st(l*blade+hum*3, r*blade+hum*3)*0.15
def aircon():
    s=lambda f:1/np.sqrt(f)*bp(80,3500)(f)
    l,r=shaped(N,s),shaped(N,s); t=np.arange(N)/SR
    hum=0.6*np.sin(2*np.pi*120*t)+0.3*np.sin(2*np.pi*240*t)+0.15*np.sin(2*np.pi*360*t+1)
    m=1+0.04*lfo(N,30)
    return st(l*m+hum*0.25, r*m+hum*0.25)*0.15
def drops(n, rate, fc_lo, fc_hi, dur_ms=(2,12), amp=(0.2,1.0), seed=3):
    r=np.random.default_rng(seed); buf=np.zeros(n); k=int(rate*n/SR)
    for p in r.integers(0,n,k):
        d=int(SR*r.uniform(*dur_ms)/1000); fc=r.uniform(fc_lo,fc_hi); t=np.arange(d)/SR
        ev=np.sin(2*np.pi*fc*t*(1+3*t))*np.exp(-t*r.uniform(300,900))*r.uniform(*amp)**2
        place(buf,ev,p)
    return buf
def rain(density=1.0, bed=0.6, low=False, seed=5):
    sb=(lambda f:1/np.sqrt(f)*bp(300,9000)(f)) if not low else (lambda f:1/np.sqrt(f)*bp(120,5000)(f))
    L=shaped(N,sb,seed)*bed; R=shaped(N,sb,seed+1)*bed
    dl=drops(N,int(900*density),1500,7000 if not low else 4000,seed=seed+2); dr=drops(N,int(900*density),1500,7000 if not low else 4000,seed=seed+3)
    dl/=np.std(dl); dr/=np.std(dr)
    sw=1+0.15*smooth_rand(N,8,seed)
    return st((L+0.5*dl)*sw,(R+0.5*dr)*sw)*0.12
def rain_soft(): return rain(0.7,0.5,seed=11)
def rain_roof():
    x=rain(1.3,0.5,low=True,seed=21); x=np.stack([filt(x[:,0],lp(3500)),filt(x[:,1],lp(3500))],1)
    return x
def rain_heavy():
    x=rain(3.0,1.2,seed=31); rumble=shaped(N,lambda f:1/f*bp(30,200)(f),77)*0.5*(0.6+0.4*smooth_rand(N,6,4))
    return x+st(rumble,rumble)*0.12
def ocean(dist=False, seed=40):
    n=N; t=np.arange(n)/n; out=[]
    period_cycles=12  # waves per 120s -> 10s
    r=np.random.default_rng(seed)
    env=np.zeros(n)
    for c in range(period_cycles):
        ph=(c+r.uniform(-.15,.15))/period_cycles; w=r.uniform(.6,1)
        d=np.minimum(np.abs(t-ph),1-np.abs(t-ph))*D  # seconds from crest (circular)
        rise=np.exp(-(d/2.4)**2); tail=np.where(((t-ph)%1)*D<9,np.exp(-((t-ph)%1)*D/3.6),0)
        env+=w*(0.6*rise+0.8*tail*(((t-ph)%1)*D>0))
    env=0.4+env/env.max()
    chans=[]
    for s in (seed,seed+1):
        base=shaped(n,lambda f:1/np.sqrt(f)*hp(40)(f),s)
        lo=filt(base,lp(500)); hi=filt(base,bp(800,8000))
        e2=np.roll(env, int(SR*0.4) if s%2 else 0)
        y=lo*(0.4+0.6*e2)+hi*e2**2*0.9
        if dist: y=filt(y,lp(900))
        chans.append(y)
    return st(*chans)*0.12
def ocean_waves(): return ocean(False,40)
def ocean_distant(): return ocean(True,50)
def underwater():
    s=lambda f:1/f*bp(20,350)(f); l,r=shaped(N,s,61),shaped(N,s,62)
    m=0.7+0.3*smooth_rand(N,6,9)
    b=np.zeros(N); rr=np.random.default_rng(63)
    for p in rr.integers(0,N,int(D*1.5)):
        d=int(SR*0.06); tt=np.arange(d)/SR; f0=rr.uniform(300,900)
        place(b,np.sin(2*np.pi*f0*tt*(1+8*tt))*np.exp(-tt*60)*rr.uniform(.1,.4),p)
    b=filt(b,lp(1500))
    return st(l*m+b*3,r*m+b*3)*0.15
def wind(center=(250,900), q=0.6, gust=10, seed=70, rustle=0.0):
    out=[]
    for k in (0,1):
        base=shaped(N,lambda f:1/np.sqrt(f)*hp(30)(f),seed+k)
        # blend of several bandpassed layers with slow crossfades -> moving resonance
        bands=np.geomspace(center[0],center[1],5); y=np.zeros(N)
        for i,fc in enumerate(bands):
            g=smooth_rand(N,gust,seed+10*i+k)**2
            y+=filt(base,lambda f,fc=fc:np.exp(-((np.log2(f/fc))**2)/(2*q**2)))*g
        amp=0.35+0.65*smooth_rand(N,gust,seed+99)
        y=y*amp
        if rustle: 
            rs=filt(shaped(N,lambda f:np.ones_like(f),seed+50+k),bp(2500,9000))*smooth_rand(N,40,seed+5+k)**3
            y=y/np.std(y)+rustle*rs/np.std(rs)
        out.append(y)
    return st(*out)*0.12
def wind_gentle(): return wind((200,700),0.5,8,70)
def wind_deep(): return wind((80,300),0.45,6,80)
def wind_trees(): return wind((300,1200),0.6,10,90,rustle=0.6)
def crickets():
    t=np.arange(N)/SR; out=[]
    bed=shaped(N,lambda f:1/f*bp(40,400)(f),101)*0.15
    for ch in (0,1):
        y=np.zeros(N); rr=np.random.default_rng(110+ch)
        for c in range(5):
            f0=rr.uniform(3900,5200); rate=rr.uniform(2.0,3.4)*D//D; per=1/rr.uniform(1.8,3.2)
            per=D/round(D/per)  # integer chirps per loop
            pulses=(np.sin(2*np.pi*t/per*0+2*np.pi*(t%per)/per*0)+0)  # placeholder
            phase=(t+rr.uniform(0,per))%per
            chirp=(phase<0.12)*np.sin(np.pi*np.clip(phase/0.12,0,1))
            trill=(np.sin(2*np.pi*55*phase)>0)*1.0
            y+=np.sin(2*np.pi*f0*t)*chirp*trill*rr.uniform(.2,1)*(0.6+0.4*lfo(N,rr.integers(1,6),rr.uniform(0,6)))
        out.append(reverb_circ(y,1.2,0.35,seed=ch)+bed*np.std(y)*3)
    return st(*out)*0.3
def clock():
    rr=np.random.default_rng(130); y=np.zeros(N)
    for i in range(D):
        d=int(SR*0.03); tt=np.arange(d)/SR; f0=2600 if i%2 else 2100
        ev=(np.sin(2*np.pi*f0*tt)+0.5*np.sin(2*np.pi*f0*2.7*tt))*np.exp(-tt*350)
        place(y,ev*0.6,int(i*SR+rr.integers(-30,30)))
    y=reverb_circ(y,0.6,0.25,2); room=shaped(N,lambda f:1/f*bp(30,300)(f),131)*0.01
    return st(y*0.5+room,y*0.5+room)
def fireplace():
    rr=np.random.default_rng(140); out=[]
    for ch in (0,1):
        y=np.zeros(N)
        for p in rr.integers(0,N,int(D*14)):
            d=int(SR*rr.uniform(.002,.02)); tt=np.arange(d)/SR
            place(y,rr.standard_normal(d)*np.exp(-tt*rr.uniform(150,600))*rr.uniform(.05,1)**2.5,p)
        for p in rr.integers(0,N,int(D*0.6)):  # pops
            d=int(SR*.05); tt=np.arange(d)/SR
            place(y,rr.standard_normal(d)*np.exp(-tt*80)*rr.uniform(.3,.7),p)
        y=filt(y,bp(400,6500))
        roar=shaped(N,lambda f:1/f*bp(40,600)(f),141+ch)*(0.6+0.4*smooth_rand(N,10,142+ch))
        out.append(y/np.std(y)*0.35+roar)
    return st(*out)*0.12
# ---------- melodies ----------
def note(m): return 440*2**((m-69)/12)
def tone(f, dur, kind='bell', vel=1.0):
    n=int(SR*dur); t=np.arange(n)/SR
    if kind=='bell':  parts=[(1,1,1.2),(2,0.35,2.2),(3,0.12,3.5),(4.2,0.06,5)]
    elif kind=='box': parts=[(1,1,1.6),(3,0.25,4),(5.4,0.08,7),(8.9,0.03,9)]
    elif kind=='piano': parts=[(1,1,0.9),(2,0.45,1.4),(3,0.2,2.0),(4,0.1,2.8),(5,0.05,3.5)]
    elif kind=='harp': parts=[(1,1,1.1),(2,0.5,1.8),(3,0.25,2.6),(4,0.12,3.4)]
    elif kind=='bowl': parts=[(1,1,0.12),(2.76,0.5,0.25),(5.40,0.25,0.4),(8.93,0.1,0.6)]
    y=np.zeros(n)
    for h,a,dec in parts:
        y+=a*np.sin(2*np.pi*f*h*t+ (0.2*np.sin(2*np.pi*0.5*t) if kind=='bowl' else 0))*np.exp(-t*dec)
    att=int(SR*(0.004 if kind!='bowl' else 0.02)); y[:att]*=np.linspace(0,1,att)
    return y*vel
def render(events, kind, n=N, t60=3.0, mix=0.35, seed=1, pan=True):
    L=np.zeros(n); R=np.zeros(n)
    for (t0,m,dur,vel) in events:
        y=tone(note(m),dur,kind,vel); p=int(t0*SR)%n
        pn=0.5+0.3*np.sin(m*1.3) if pan else 0.5
        place(L,y*np.sqrt(1-pn),p); place(R,y*np.sqrt(pn),p)
    return st(reverb_circ(L,t60,mix,seed),reverb_circ(R,t60,mix,seed+1))
def pad(chords, n=N, seed=1):
    t=np.arange(n)/SR; L=np.zeros(n); R=np.zeros(n); seg=n//len(chords)
    for i,ch in enumerate(chords):
        env=np.zeros(n); c0=i*seg
        idx=np.arange(n); d=((idx-c0)%n)
        w=np.where(d<seg*1.5, np.sin(np.pi*np.clip(d/(seg*1.5),0,1))**2,0)
        for m in ch:
            f=note(m)
            for det,side in ((-0.15,0),(0.15,1)):
                ff=f*2**(det/12/10)
                v=sum(np.sin(2*np.pi*ff*k*t)/k**1.6 for k in range(1,6))
                (L if side==0 else R)[:]+=v*w*0.3
    L=filt(L,lp(1800)); R=filt(R,lp(1800))
    return st(reverb_circ(L,4,0.5,seed),reverb_circ(R,4,0.5,seed+1))
PENTA=[0,2,4,7,9]
def scale_notes(root, lo, hi, sc=PENTA): return [root+o*12+s for o in range(-3,4) for s in sc if lo<=root+o*12+s<=hi]
def music_box():
    rr=np.random.default_rng(201); ns=scale_notes(72,67,91); ev=[]; beat=0.75; nb=int(D/beat)
    i=len(ns)//2
    for b in range(nb):
        if rr.random()<0.72:
            i=int(np.clip(i+rr.choice([-2,-1,1,1,2,0]),0,len(ns)-1)); ev.append((b*beat,ns[i],3.0,rr.uniform(.5,.9)))
        if b%8==0: ev.append((b*beat,ns[0]-12,4.0,0.4))
    return render(ev,'box',t60=2.2,mix=0.35,seed=3)*0.25
def soft_piano():
    prog=[[60,64,67,71],[57,60,64,67],[53,57,60,64],[55,59,62,67]]; ev=[]; chord_len=7.5; nb=int(D/chord_len)
    rr=np.random.default_rng(202)
    for c in range(nb):
        ch=prog[c%4]; t0=c*chord_len
        ev.append((t0,ch[0]-12,6,0.5))
        for k,m in enumerate([ch[1],ch[2],ch[3],ch[2]+12, ch[3], ch[1]+12]):
            ev.append((t0+0.6+k*1.05+rr.uniform(-.03,.03),m,4,rr.uniform(.3,.55)))
    return render(ev,'piano',t60=3.2,mix=0.4,seed=5)*0.22
def drone_pad():
    return pad([[48,55,60,64],[45,52,57,60],[41,48,53,57],[43,50,55,62]],seed=7)*0.12
def harp():
    rr=np.random.default_rng(204); ev=[]; prog=[[62,66,69,74],[59,62,66,71],[55,59,62,67],[57,61,64,69]]
    cl=6.0
    for c in range(int(D/cl)):
        ch=prog[c%4]; up=ch+[m+12 for m in ch]
        for k,m in enumerate(up if c%2==0 else up[::-1]):
            ev.append((c*cl+k*0.32,m,3.5,rr.uniform(.35,.6)))
        ev.append((c*cl,ch[0]-12,5,0.5))
    h=render(ev,'harp',t60=4.5,mix=0.5,seed=9)*0.2
    p=pad([[50,57,62,66],[47,54,59,62],[43,50,55,59],[45,52,57,61]]*5,seed=17)*0.035
    return h+p
def singing_bowls():
    rr=np.random.default_rng(205); ev=[]; tones=[50,57,62,55]
    for k in range(10):
        ev.append((k*12+rr.uniform(0,1),tones[k%4],20,rr.uniform(.5,.8)))
    return render(ev,'bowl',t60=5,mix=0.45,seed=11)*0.25
def dream_bells():
    rr=np.random.default_rng(206); ns=scale_notes(69,64,88); ev=[]; beat=1.5; i=5
    for b in range(int(D/beat)):
        if rr.random()<0.55:
            i=int(np.clip(i+rr.choice([-1,1,-2,2]),0,len(ns)-1)); ev.append((b*beat,ns[i],5,rr.uniform(.4,.7)))
    bells=render(ev,'bell',t60=4,mix=0.5,seed=13)*0.2
    p=pad([[45,52,57,64],[41,48,53,60],[43,50,55,62],[40,47,52,59]],seed=15)*0.05
    return bells+p
ALL=dict(white=white,pink=pink,brown=brown,deep_brown=deep_brown,grey=grey,green=green,fan=fan,aircon=aircon,
 rain_soft=rain_soft,rain_roof=rain_roof,rain_heavy=rain_heavy,ocean_waves=ocean_waves,ocean_distant=ocean_distant,
 underwater=underwater,wind_gentle=wind_gentle,wind_deep=wind_deep,wind_trees=wind_trees,crickets=crickets,clock=clock,
 fireplace=fireplace,music_box=music_box,soft_piano=soft_piano,drone_pad=drone_pad,harp=harp,singing_bowls=singing_bowls,dream_bells=dream_bells)
if __name__=='__main__':
    names=sys.argv[1:] or list(ALL)
    for k in names:
        x=ALL[k](); x=x/np.max(np.abs(x))*0.7
        sf.write(f'gen/{k}.wav',x.astype(np.float32),SR,subtype='FLOAT'); print(k,x.shape,flush=True)
