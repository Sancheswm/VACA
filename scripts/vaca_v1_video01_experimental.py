"""VACA V1 experimental baseline for CattleEyeView video 01.

Engineering-only fallback. It deliberately does not emit lameness, BCS or cow-ID
claims. Production replacements are RTMDet-Ins -> BoT-SORT -> open-set ReID ->
RTMPose-Bovine15 -> temporal mobility model.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math
from pathlib import Path
import cv2
import numpy as np

STATUS = "NOT_CLINICALLY_VALIDATED"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def background(video: str, size: tuple[int, int], n: int = 20) -> np.ndarray:
    cap = cv2.VideoCapture(video); xs = []
    for _ in range(n):
        ok, fr = cap.read()
        if not ok: break
        fr = cv2.resize(fr, size, interpolation=cv2.INTER_AREA)
        xs.append(cv2.cvtColor(fr, cv2.COLOR_BGR2LAB).astype(np.float32))
    cap.release()
    if not xs: raise RuntimeError("no frames for background")
    return np.median(np.stack(xs), axis=0).astype(np.float32)


def roi_masks(h: int, w: int):
    roi = np.zeros((h, w), np.uint8)
    cv2.rectangle(roi, (int(.0625*w), int(.167*h)), (int(.9375*w), int(.87*h)), 255, -1)
    ref = np.zeros((h, w), bool)
    ref[int(.28*h):int(.74*h), int(.13*w):int(.86*w)] = True
    return roi, ref


def segment(fr: np.ndarray, bg: np.ndarray, roi: np.ndarray, ref: np.ndarray, thr=30.) -> np.ndarray:
    lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = lab - bg
    d[:, :, 0] -= np.median(d[:, :, 0][ref])
    score = .6*np.abs(d[:, :, 0]) + .2*np.abs(d[:, :, 1]) + .2*np.abs(d[:, :, 2])
    m = ((score > thr) & (roi > 0)).astype(np.uint8)*255
    m = cv2.medianBlur(m, 5)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def objects(mask: np.ndarray, scale: float):
    out = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    amin, agroup = max(300., 8000*scale*scale), max(6000., 180000*scale*scale)
    for c in contours:
        a = float(cv2.contourArea(c))
        if a < amin: continue
        mm = cv2.moments(c)
        if not mm["m00"]: continue
        cx, cy = mm["m10"]/mm["m00"], mm["m01"]/mm["m00"]
        r = cv2.minAreaRect(c); rw, rh = r[1]; ang = float(r[2]) + (90. if rw < rh else 0.)
        out.append(dict(contour=c, area=a, centroid=(cx, cy), rect=r, angle=ang,
                        major=max(rw, rh), group=a > agroup))
    return sorted(out, key=lambda x: x["area"], reverse=True)


class Tracker:
    def __init__(self, max_d=90., max_missed=8):
        self.max_d, self.max_missed, self.next_id = max_d, max_missed, 1
        self.t = {}; self.closed = []
    def update(self, dets, frame):
        if not self.t:
            return {i: self._new(d, frame) for i, d in enumerate(dets)}
        ids = list(self.t); used_t=set(); used_d=set(); mapping={}
        pairs=[]
        for i, tid in enumerate(ids):
            for j, d in enumerate(dets):
                pairs.append((np.hypot(self.t[tid]["xy"][0]-d["centroid"][0], self.t[tid]["xy"][1]-d["centroid"][1]), tid, j))
        for dist, tid, j in sorted(pairs):
            if dist > self.max_d: break
            if tid in used_t or j in used_d: continue
            d=dets[j]; tr=self.t[tid]; tr["xy"]=d["centroid"]; tr["missed"]=0
            tr["hist"].append((frame, *d["centroid"], d["area"], d["angle"]))
            mapping[j]=tid; used_t.add(tid); used_d.add(j)
        for tid in ids:
            if tid not in used_t: self.t[tid]["missed"] += 1
        for j,d in enumerate(dets):
            if j not in used_d: mapping[j]=self._new(d,frame)
        for tid in [k for k,v in self.t.items() if v["missed"] > self.max_missed]:
            self.closed.append(self.t.pop(tid))
        return mapping
    def _new(self,d,frame):
        tid=self.next_id; self.next_id+=1
        self.t[tid]={"id":tid,"xy":d["centroid"],"missed":0,"hist":[(frame,*d["centroid"],d["area"],d["angle"])]}
        return tid


def yolo_union(path: Path, w: int, h: int):
    m=np.zeros((h,w),np.uint8)
    for line in path.read_text().splitlines():
        v=line.split(); z=list(map(float,v[1:])); pts=np.array([(int(z[i]*w),int(z[i+1]*h)) for i in range(0,len(z)-1,2)],np.int32)
        if len(pts)>=3: cv2.fillPoly(m,[pts],255)
    return m


def validate(video, bg, roi, ref, labels_dir):
    p=Path(labels_dir) if labels_dir else None
    labels=sorted(p.glob("*.txt")) if p and p.exists() else []
    if not labels: return {"status":"NOT_RUN"}
    h,w=bg.shape[:2]; cap=cv2.VideoCapture(video); rows=[]
    for lab in labels:
        idx=int(lab.stem); cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,fr=cap.read()
        if not ok: continue
        fr=cv2.resize(fr,(w,h),interpolation=cv2.INTER_AREA); pr=segment(fr,bg,roi,ref)>0; gt=yolo_union(lab,w,h)>0
        inter=(pr&gt).sum(); union=(pr|gt).sum(); pa=pr.sum(); ga=gt.sum()
        rows.append(dict(frame=idx,iou=float(inter/union) if union else 1.,precision=float(inter/pa) if pa else 0.,recall=float(inter/ga) if ga else 0.,gt_fraction=float(ga/(w*h))))
    cap.release(); substantial=[r for r in rows if r["gt_fraction"]>=.01]
    mean=lambda k,a: float(np.mean([r[k] for r in a])) if a else None
    return dict(status="EXPERIMENTAL_CLASSICAL_BASELINE",frames=rows,mean_iou_all=mean("iou",rows),
                mean_precision_all=mean("precision",rows),mean_recall_all=mean("recall",rows),
                mean_iou_gt_ge_1pct=mean("iou",substantial),n_frames=len(rows))


def run(video: str, out_dir: str, labels_dir=None, width=480):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(video); fps=float(cap.get(cv2.CAP_PROP_FPS) or 8); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    h=round(H*width/W); bg=background(video,(width,h)); roi,ref=roi_masks(h,width); roi_area=(roi>0).sum(); scale=width/1920.
    paths={k:out/f"VACA_V1_EXP_video01_{v}" for k,v in {"video":"overlay.mp4","csv":"telemetry.csv","summary":"summary.json","validation":"validation.json","manifest":"manifest.json"}.items()}
    wr=cv2.VideoWriter(str(paths["video"]),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,h)); cap=cv2.VideoCapture(video); tr=Tracker(max_d=90*scale/.25)
    rows=[]; occs=[]; blurs=[]; brights=[]; events=[]; active=False; start=0; mx=0.; low=0; maxcand=0; i=0
    while True:
        ok,fr=cap.read()
        if not ok: break
        fr=cv2.resize(fr,(width,h),interpolation=cv2.INTER_AREA); m=segment(fr,bg,roi,ref); det=objects(m,scale); ids=tr.update(det,i); occ=float((m>0).sum()/roi_area); occs.append(occ); maxcand=max(maxcand,len(det))
        if i % max(1,round(fps)) == 0:
            g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY); blurs.append(float(cv2.Laplacian(g,cv2.CV_64F).var())); brights.append(float(g[roi>0].mean()))
        if not active and occ>=.03: active=True; start=i; mx=occ; low=0
        elif active:
            mx=max(mx,occ); low=low+1 if occ<.02 else 0
            if low>=round(fps):
                end=i-round(fps); events.append(dict(event_id=len(events)+1,start_frame=start,end_frame=end,start_s=start/fps,end_s=end/fps,duration_s=(end-start+1)/fps,max_foreground_occupancy=mx,semantic="PASSAGE_GROUP_NOT_INDIVIDUAL_COW")); active=False; low=0
        vis=fr.copy(); ov=vis.copy(); ov[m>0]=(70,190,70); vis=cv2.addWeighted(ov,.35,vis,.65,0)
        cv2.rectangle(vis,(int(.0625*width),int(.167*h)),(int(.9375*width),int(.87*h)),(255,220,0),2)
        for j,d in enumerate(det):
            tid=ids.get(j,-1); box=cv2.boxPoints(d["rect"]).astype(np.int32); cv2.polylines(vis,[box],True,(0,255,255),2); cx,cy=map(int,d["centroid"]); a=math.radians(d["angle"]); half=max(15.,d["major"]*.42); p1=(int(cx-math.cos(a)*half),int(cy-math.sin(a)*half)); p2=(int(cx+math.cos(a)*half),int(cy+math.sin(a)*half)); cv2.line(vis,p1,p2,(255,0,255),2); cv2.circle(vis,(cx,cy),4,(0,0,255),-1); cv2.putText(vis,f"T{tid} {'GROUP' if d['group'] else 'OBJ'}",(max(3,cx-45),max(18,cy-12)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1,cv2.LINE_AA)
            rows.append(dict(frame=i,time_s=i/fps,track_id=tid,cx=d["centroid"][0],cy=d["centroid"][1],area_px=d["area"],major_axis_deg=d["angle"],is_group_candidate=int(d["group"]),occupancy=occ,status=STATUS))
        cv2.rectangle(vis,(0,0),(width,62),(0,0,0),-1); cv2.putText(vis,"VACA V1 EXP | VIDEO 01",(12,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2,cv2.LINE_AA); cv2.putText(vis,"ENGINEERING ONLY",(330,22),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,180,255),1,cv2.LINE_AA); cv2.putText(vis,f"MotionSeg fallback | candidates={len(det)} | occupancy={occ:.3f} | t={i/fps:.1f}s",(12,47),cv2.FONT_HERSHEY_SIMPLEX,.42,(180,255,180),1,cv2.LINE_AA); wr.write(vis); i+=1
    cap.release(); wr.release()
    if active: events.append(dict(event_id=len(events)+1,start_frame=start,end_frame=i-1,start_s=start/fps,end_s=(i-1)/fps,duration_s=(i-start)/fps,max_foreground_occupancy=mx,semantic="PASSAGE_GROUP_NOT_INDIVIDUAL_COW"))
    with open(paths["csv"],"w",newline="") as f:
        fw=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ["frame"]); fw.writeheader(); fw.writerows(rows)
    val=validate(video,bg,roi,ref,labels_dir); paths["validation"].write_text(json.dumps(val,indent=2))
    summary=dict(experiment="VACA_V1_EXP_VIDEO01",status=STATUS,input=dict(video=Path(video).name,frames=n,fps=fps,width=W,height=H,duration_s=n/fps,processing_width=width,processing_height=h),implemented_stages=dict(capture_quality_gate="ACTIVE_CLASSICAL",detection_segmentation="MOTION_SEGMENTATION_FALLBACK",tracking="CENTROID_GREEDY_FALLBACK",cow_reid="NOT_IMPLEMENTED",bovine15_pose="NOT_IMPLEMENTED",temporal_mobility_model="BLOCKED",bcs="NOT_IMPLEMENTED"),capture_quality=dict(median_laplacian_variance=float(np.median(blurs)),mean_roi_brightness=float(np.mean(brights)),std_roi_brightness=float(np.std(brights))),foreground=dict(mean_occupancy=float(np.mean(occs)),max_occupancy=float(np.max(occs)),max_simultaneous_component_candidates=maxcand),passage_groups=events,segmentation_validation=val,clinical_interpretation="NONE",next_production_replacement="RTMDet-Ins -> BoT-SORT -> ReID -> RTMPose-Bovine15")
    paths["summary"].write_text(json.dumps(summary,indent=2)); manifest=dict(experiment_id="VACA_V1_EXP_VIDEO01_2026-09-11",input_drive_file_id="1FgAckqvHJKvbw_dpnHMTMD1vVtCLUNga",input_sha256=sha256(video),source_dataset="CattleEyeView",clinical_status=STATUS,artifacts=[p.name for p in paths.values()],limitations=["temporary motion segmentation","provisional non-identity tracks","no Bovine15/ReID/BCS/clinical model"]); paths["manifest"].write_text(json.dumps(manifest,indent=2)); return summary


def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--labels-dir"); p.add_argument("--width",type=int,default=480); a=p.parse_args(); print(json.dumps(run(a.input,a.output_dir,a.labels_dir,a.width),indent=2))


if __name__ == "__main__": main()
