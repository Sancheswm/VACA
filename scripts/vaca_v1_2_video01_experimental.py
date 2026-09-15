import cv2, numpy as np, pandas as pd, json, os, time
from sklearn.ensemble import ExtraTreesClassifier
from scipy.optimize import linear_sum_assignment
VIDEO='/mnt/data/01_960.mp4'; OUT='/mnt/data/vaca_v11_excellence'; os.makedirs(OUT,exist_ok=True)
TRAIN_FRAMES=[1144,1156,1172]; HOLDOUT_FRAMES=[1150,1167]; ALL_FRAMES=[1144,1150,1156,1167,1172]
LABELS={1144:'/mnt/data/01144.txt',1150:'/mnt/data/01150.txt',1156:'/mnt/data/01156.txt',1167:'/mnt/data/01167.txt',1172:'/mnt/data/01172.txt'}
IMAGES={i:f'/mnt/data/frame_{i:04d}.jpg' for i in ALL_FRAMES}
S=2; TH=.75; RNG=np.random.default_rng(42)
cap=cv2.VideoCapture(VIDEO);N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));FPS=float(cap.get(cv2.CAP_PROP_FPS));W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));w,h=W//S,H//S
import glob
lab_samples=[]
for fp in sorted(glob.glob('/mnt/data/bg_samples/*.jpg')):
 im=cv2.imread(fp)
 if im is not None:
  lab_samples.append(cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(np.float32))
cap.release(); st=np.stack(lab_samples); BG=np.median(st,0); MAD=np.median(np.abs(st-BG),0)+3.; YG,XG=np.mgrid[0:h,0:w].astype(np.float32); XN=XG/w;YN=YG/h;ROI=(YN>.27)&(YN<.73)
def feat(im):
 lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(np.float32);hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV).astype(np.float32);d=np.abs(lab-BG);z=d/MAD;gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32);mu=cv2.GaussianBlur(gray,(0,0),2);std=np.sqrt(np.maximum(cv2.GaussianBlur(gray*gray,(0,0),2)-mu*mu,0));return np.dstack([lab/255,hsv/255,d/255,z/10,np.linalg.norm(d,axis=2)[...,None]/255,std[...,None]/80,XN[...,None],YN[...,None]])
def gt(path):
 m=np.zeros((H,W),np.uint8)
 for l in open(path):
  a=list(map(float,l.split()))
  if len(a)<7:continue
  pts=np.array([(a[k]*W,a[k+1]*H) for k in range(1,len(a),2)],np.int32);cv2.fillPoly(m,[pts],255)
 return cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST)
def load_small(frame_idx):
 fr=cv2.imread(IMAGES[frame_idx]); return cv2.resize(fr,(w,h),interpolation=cv2.INTER_AREA)
X=[];Y=[]
for fi in TRAIN_FRAMES:
 im=load_small(fi);F=feat(im);G=gt(LABELS[fi])>0;p=np.argwhere(G&ROI);n=np.argwhere((~G)&ROI);p=p[RNG.choice(len(p),min(12000,len(p)),False)];n=n[RNG.choice(len(n),min(18000,len(n)),False)];c=np.vstack([p,n]);X.append(F[c[:,0],c[:,1]]);Y.append(np.r_[np.ones(len(p),np.uint8),np.zeros(len(n),np.uint8)])
clf=ExtraTreesClassifier(n_estimators=24,max_depth=14,min_samples_leaf=4,n_jobs=-1,class_weight='balanced',random_state=42,max_features=.7).fit(np.vstack(X),np.concatenate(Y))
def prob(im):
 F=feat(im);p=np.zeros((h,w),np.float32);p[ROI]=clf.predict_proba(F[ROI])[:,1];return p
def clean(p):
 m=(p>TH).astype(np.uint8)*255;m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8));m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8));n,la,st,_=cv2.connectedComponentsWithStats(m);o=np.zeros_like(m)
 for j in range(1,n):
  if st[j,cv2.CC_STAT_AREA]>=30:o[la==j]=255
 return o
def metric(p,g):
 P=p>0;G=g>0;i=(P&G).sum();u=(P|G).sum();fp=(P&~G).sum();fn=(~P&G).sum();return {'iou':float(i/(u+1e-9)),'precision':float(i/(i+fp+1e-9)),'recall':float(i/(i+fn+1e-9))}
val=[]
for fi in ALL_FRAMES:
 q=metric(clean(prob(load_small(fi))),gt(LABELS[fi]));q.update(frame=fi,split='train' if fi in TRAIN_FRAMES else 'holdout');val.append(q)
print('validation',val,flush=True)
def instances(m):
 n,la,st,_=cv2.connectedComponentsWithStats(m);out=[]
 for j in range(1,n):
  area=int(st[j,cv2.CC_STAT_AREA])
  if area<45:continue
  mm=(la==j).astype(np.uint8)
  out.append(mm)
 return sorted(out,key=lambda mm:int(mm.sum()),reverse=True)[:8]
def cent(m):y,x=np.nonzero(m);return np.array([x.mean(),y.mean()],np.float32)
def bbox(m):y,x=np.nonzero(m);return(int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1)
def biou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]);ii=max(0,x2-x1)*max(0,y2-y1);aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1]);return ii/(aa+bb-ii+1e-9)
tracks={};NEXT=1
def assign(ds):
 global NEXT
 for t in tracks.values():t['age']+=1
 ids=[i for i,t in tracks.items() if t['age']<=8];used=set()
 if ids and ds:
  C=np.zeros((len(ids),len(ds)),np.float32)
  for i,tid in enumerate(ids):
   t=tracks[tid];pred=t['c']+t['v']
   for j,d in enumerate(ds):C[i,j]=.8*np.linalg.norm(pred-d['c'])/75+.2*(1-biou(t['b'],d['b']))
  rr,cc=linear_sum_assignment(C)
  for i,j in zip(rr,cc):
   if C[i,j]>1.5:continue
   tid=ids[i];t=tracks[tid];v=ds[j]['c']-t['c'];t['v']=.65*t['v']+.35*v;t['c']=ds[j]['c'];t['b']=ds[j]['b'];t['age']=0;ds[j]['tid']=tid;used.add(j)
 for j,d in enumerate(ds):
  if j not in used:tracks[NEXT]={'c':d['c'],'b':d['b'],'v':np.zeros(2,np.float32),'age':0};d['tid']=NEXT;NEXT+=1
 for i in list(tracks):
  if tracks[i]['age']>16:del tracks[i]
 return ds
def mesh(m,v):
 yy,xx=np.nonzero(m)
 if len(xx)<50:return [],[]
 P=np.c_[xx,yy].astype(np.float32);c=P.mean(0);Q=P-c;e,V=np.linalg.eigh(np.cov(Q.T));maj=V[:,np.argmax(e)];minor=np.array([-maj[1],maj[0]],np.float32)
 if np.linalg.norm(v)>.25:
  if np.dot(maj,v)<0:maj=-maj;minor=-minor
 elif maj[0]<0:maj=-maj;minor=-minor
 u=Q@maj;vv=Q@minor;lo,hi=np.percentile(u,[4,96]);us=np.linspace(lo,hi,6);nodes=[];L=[];C=[];R=[]
 for uu in us:
  band=np.abs(u-uu)<max(2.5,(hi-lo)/16);vals=np.percentile(vv[band] if band.sum()>5 else vv,[12,50,88]);L.append(len(nodes));nodes.append(c+maj*uu+minor*vals[0]);C.append(len(nodes));nodes.append(c+maj*uu+minor*vals[1]);R.append(len(nodes));nodes.append(c+maj*uu+minor*vals[2])
 E=[]
 for a in [L,C,R]:E += [(a[i],a[i+1]) for i in range(5)]
 for i in range(6):E += [(L[i],C[i]),(C[i],R[i])]
 for i in range(5):E += [(L[i],C[i+1]),(R[i],C[i+1])]
 return nodes,E

import argparse, subprocess
ap=argparse.ArgumentParser(); ap.add_argument('--start',type=int,required=True); ap.add_argument('--end',type=int,required=True); ap.add_argument('--state',required=True); ap.add_argument('--out',required=True); ap.add_argument('--telemetry',required=True); args=ap.parse_args()
tracks={}; NEXT=1
if os.path.exists(args.state):
    s=json.load(open(args.state)); NEXT=int(s.get('NEXT',1))
    for k,v in s.get('tracks',{}).items():
        tracks[int(k)]={'c':np.array(v['c'],np.float32),'b':tuple(v['b']),'v':np.array(v['v'],np.float32),'age':int(v['age'])}
cap=cv2.VideoCapture(VIDEO); cap.set(cv2.CAP_PROP_POS_FRAMES,args.start)
ff=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','ultrafast','-crf','22','-pix_fmt','yuv420p',args.out],stdin=subprocess.PIPE)
palette=[(0,215,255),(50,245,110),(255,120,50),(220,90,255),(255,220,60),(60,170,255),(255,80,170),(150,255,70)];tele=[]; idx=args.start; t0=time.time()
while idx<=args.end:
 ok,fr=cap.read()
 if not ok: break
 sm=cv2.resize(fr,(w,h),interpolation=cv2.INTER_AREA);p=prob(sm);mk=clean(p);ds=[]
 for m in instances(mk):ds.append({'m':m,'c':cent(m),'b':bbox(m),'area':int(m.sum()),'conf':float(p[m>0].mean())})
 ds=assign(ds);vis=fr.copy();ov=vis.copy()
 for d in ds:
  col=palette[(d['tid']-1)%len(palette)];mf=cv2.resize(d['m'],(W,H),interpolation=cv2.INTER_NEAREST)>0;ov[mf]=col
 vis=cv2.addWeighted(vis,.72,ov,.28,0)
 for d in ds:
  tid=d['tid'];col=palette[(tid-1)%len(palette)];tr=tracks[tid];yy,xx=np.nonzero(d['m']);P=np.c_[xx*S,yy*S].astype(np.float32);box=cv2.boxPoints(cv2.minAreaRect(P)).astype(np.int32);cv2.polylines(vis,[box],True,col,2,cv2.LINE_AA);nd,ed=mesh(d['m'],tr['v']);nd=[tuple(np.round(x*S).astype(int)) for x in nd]
  for a,b in ed:cv2.line(vis,nd[a],nd[b],(245,245,245),1,cv2.LINE_AA)
  for pt in nd:cv2.circle(vis,pt,3,col,-1,cv2.LINE_AA);cv2.circle(vis,pt,1,(255,255,255),-1,cv2.LINE_AA)
  cx,cy=(d['c']*S).astype(int);bx=max(5,min(W-220,cx-95));by=max(60,cy-48);cv2.rectangle(vis,(bx,by-29),(bx+210,by+24),(8,8,8),-1);bw=d['b'][2]-d['b'][0]; group=(d['area']>15000 or bw>250); label=(f'GROUP {tid:03d}' if group else f'COW TRACK {tid:03d}');cv2.putText(vis,label,(bx+7,by-8),cv2.FONT_HERSHEY_SIMPLEX,.50,col,2,cv2.LINE_AA);cv2.putText(vis,f'SEG {d["conf"]:.2f}  '+('REVIEW' if group else 'MOBILITY PENDING'),(bx+7,by+13),cv2.FONT_HERSHEY_SIMPLEX,.34,(240,240,240),1,cv2.LINE_AA);tele.append({'frame':idx,'time_s':idx/FPS,'track_id':tid,'cx':cx,'cy':cy,'seg_conf':d['conf'],'area_px':d['area']*S*S,'speed_px_s':float(np.linalg.norm(tr['v'])*S*FPS),'group_review':group})
 cv2.rectangle(vis,(12,12),(690,92),(5,5,5),-1);cv2.putText(vis,'VACA V1.2 - CLEAN EXPERIMENT',(24,40),cv2.FONT_HERSHEY_SIMPLEX,.72,(245,245,245),2,cv2.LINE_AA);cv2.putText(vis,'SEGMENTATION + CONSERVATIVE TRACKING + BODY MESH',(24,63),cv2.FONT_HERSHEY_SIMPLEX,.42,(205,225,225),1,cv2.LINE_AA);cv2.putText(vis,'NOT CLINICALLY VALIDATED | GEOMETRIC MESH != BOVINE15',(24,83),cv2.FONT_HERSHEY_SIMPLEX,.38,(170,195,255),1,cv2.LINE_AA);cv2.putText(vis,f'{idx/FPS:06.1f}s | objects {len(ds)}',(W-200,34),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA);ff.stdin.write(vis.tobytes());idx+=1
cap.release(); ff.stdin.close(); ff.wait(); pd.DataFrame(tele).to_csv(args.telemetry,index=False)
s={'NEXT':NEXT,'tracks':{str(k):{'c':v['c'].tolist(),'b':list(v['b']),'v':v['v'].tolist(),'age':v['age']} for k,v in tracks.items()}}; json.dump(s,open(args.state,'w'),indent=2)
print(json.dumps({'start':args.start,'end':idx-1,'frames':idx-args.start,'seconds':time.time()-t0,'NEXT':NEXT}),flush=True)
