from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Immutable project constants
# -----------------------------
CATTLEEYE_DATASET_ROOT_ID = "1Pjt1bEZR5s-grzMqnuOEpIv9A7JK1VoL"
CATTLEEYE_SEGMENT_LABELS_ID = "1x1gMFYPol99qpL5Dz3bukARXDaQvgfEm"
CATTLEEYE_DETECT_IMAGES_ID = "16COroSFlEt1uFAwJhWlnpmdCK9EeT7KH"
CATTLEEYE_VIDEOS_ID = "1_vUuM3_5evffsszdOcnEoOoc1H9j2F7Y"
CATTLEEYE_VIDEO01_ID = "1FgAckqvHJKvbw_dpnHMTMD1vVtCLUNga"
VIDEO01_SHA256 = "e80374a4659994b2361afbe04bc042ba8967fd98c2e11ff89611456d34014568"

OFFICIAL_TRAIN_VIDEOS = ("02.mp4", "03.mp4", "04.mp4", "06.mp4", "08.mp4", "11.mp4", "12.mp4", "13.mp4", "14.mp4")
# CattleEyeView README says official test and val contain the same image set.
# VACA therefore partitions the held-out videos itself to prevent leakage.
VACA_VALID_VIDEOS = ("05.mp4", "09.mp4")
VACA_TEST_VIDEOS = ("01.mp4", "07.mp4", "10.mp4")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: Path | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ffprobe_video(path: Path | str) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    meta["duration_sec"] = meta["frames"] / meta["fps"] if meta["fps"] else None
    cap.release()
    return meta


# -----------------------------
# Google Drive API helpers
# -----------------------------
def build_drive_service():
    """Authenticate inside Colab and return Drive v3 service (fallback only).

    Normal VACA Colab runs should prefer mounted shortcut paths; this function
    is only needed if the mounted dataset cannot be found.
    """
    try:
        from google.colab import auth
    except ImportError as exc:
        raise RuntimeError("build_drive_service() must run inside Google Colab") from exc
    auth.authenticate_user()
    from googleapiclient.discovery import build
    return build("drive", "v3", cache_discovery=False)


def drive_list_children(service, folder_id: str) -> list[dict]:
    out: list[dict] = []
    token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id,name,mimeType,size,md5Checksum)",
            pageSize=1000,
            pageToken=token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        out.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return out


def drive_find_child(service, folder_id: str, name: str) -> Optional[dict]:
    for item in drive_list_children(service, folder_id):
        if item.get("name") == name:
            return item
    return None


def drive_download_file(service, file_id: str, dest: Path | str, expected_size: Optional[int] = None) -> Path:
    from googleapiclient.http import MediaIoBaseDownload
    dest = Path(dest)
    ensure_dir(dest.parent)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return dest
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"  {dest.name}: {status.progress()*100:5.1f}%", end="\r")
    tmp.replace(dest)
    print(f"  {dest.name}: OK{' ' * 20}")
    return dest


def drive_mirror_folder(service, folder_id: str, dest_dir: Path | str, *, skip_existing: bool = True) -> Path:
    """Recursively mirror an accessible Drive folder. Safe to rerun."""
    dest_dir = ensure_dir(dest_dir)
    stack: list[tuple[str, Path]] = [(folder_id, dest_dir)]
    total_files = 0
    while stack:
        current_id, current_dir = stack.pop()
        ensure_dir(current_dir)
        items = drive_list_children(service, current_id)
        for item in items:
            name = item["name"]
            mime = item.get("mimeType", "")
            if mime == "application/vnd.google-apps.folder":
                stack.append((item["id"], current_dir / name))
            else:
                dest = current_dir / name
                size = int(item["size"]) if item.get("size") else None
                if skip_existing and dest.exists() and (size is None or dest.stat().st_size == size):
                    continue
                drive_download_file(service, item["id"], dest, size)
                total_files += 1
                if total_files % 100 == 0:
                    print(f"Downloaded {total_files} files into {dest_dir}")
    return dest_dir


def drive_download_named_video(service, videos_folder_id: str, video_name: str, dest: Path | str) -> Path:
    item = drive_find_child(service, videos_folder_id, video_name)
    if item is None:
        raise FileNotFoundError(f"{video_name} not found in Drive folder {videos_folder_id}")
    if item.get("mimeType") == "application/vnd.google-apps.folder":
        raise RuntimeError(f"Expected video file but got folder: {video_name}")
    size = int(item["size"]) if item.get("size") else None
    return drive_download_file(service, item["id"], dest, size)


def limited_find_dirs(root: Path, suffix_parts: tuple[str, ...], max_depth: int = 6) -> list[Path]:
    root = Path(root)
    targets: list[Path] = []
    base_depth = len(root.parts)
    for current, dirs, _ in os.walk(root):
        p = Path(current)
        depth = len(p.parts) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        if tuple(p.parts[-len(suffix_parts):]) == suffix_parts:
            targets.append(p)
            dirs[:] = []
    return targets


def _mounted_dataset_roots(my_drive: Path = Path("/content/drive/MyDrive")) -> list[Path]:
    shortcut_target = Path("/content/drive/.shortcut-targets-by-id/1Pjt1bEZR5s-grzMqnuOEpIv9A7JK1VoL/dataset")
    roots = [
        shortcut_target,
        my_drive / "dataset",
        my_drive / "CattleEyeView",
        my_drive / "CattleEyeView" / "dataset",
    ]
    out = []
    seen = set()
    for p in roots:
        key = str(p)
        if key not in seen:
            out.append(p)
            seen.add(key)
    return out


def discover_mounted_cattleeye(my_drive: Path = Path("/content/drive/MyDrive")) -> tuple[Optional[Path], Optional[Path]]:
    image_candidates = [r / "annotation" / "detect" / "images" for r in _mounted_dataset_roots(my_drive)]
    label_candidates = [r / "annotation" / "segment" / "labels" for r in _mounted_dataset_roots(my_drive)]
    images = next((p for p in image_candidates if p.exists()), None)
    labels = next((p for p in label_candidates if p.exists()), None)
    if images is None:
        found = limited_find_dirs(my_drive, ("annotation", "detect", "images"), max_depth=6)
        images = found[0] if found else None
    if labels is None:
        found = limited_find_dirs(my_drive, ("annotation", "segment", "labels"), max_depth=6)
        labels = found[0] if found else None
    return images, labels


def discover_mounted_videos(my_drive: Path = Path("/content/drive/MyDrive")) -> Optional[Path]:
    candidates = [r / "videos" for r in _mounted_dataset_roots(my_drive)]
    videos = next((p for p in candidates if p.exists()), None)
    if videos is None:
        found = limited_find_dirs(my_drive, ("videos",), max_depth=5)
        videos = next((p for p in found if (p / "01.mp4").exists()), None)
    return videos


# -----------------------------
# CattleEyeView -> COCO-instance
# -----------------------------
def _polygon_area(points: list[float]) -> float:
    xy = list(zip(points[0::2], points[1::2]))
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(xy, xy[1:] + xy[:1]))) / 2.0


def _bbox(points: list[float]) -> list[float]:
    xs, ys = points[0::2], points[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]


def parse_yolo_polygon_label(label_path: Path, width: int, height: int, class_id: int = 0) -> list[dict]:
    anns = []
    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        toks = raw.split()
        try:
            cls = int(float(toks[0]))
            coords = [float(v) for v in toks[1:]]
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_no}: malformed label") from exc
        if cls != class_id:
            continue
        if len(coords) < 6 or len(coords) % 2:
            raise ValueError(f"{label_path}:{line_no}: polygon needs >=3 xy pairs")
        pix: list[float] = []
        for i in range(0, len(coords), 2):
            x = min(1.0, max(0.0, coords[i])) * width
            y = min(1.0, max(0.0, coords[i + 1])) * height
            pix.extend([float(x), float(y)])
        area = _polygon_area(pix)
        if not math.isfinite(area) or area <= 1.0:
            continue
        anns.append({"segmentation": [pix], "bbox": _bbox(pix), "area": float(area)})
    return anns


def _image_index(video_image_dir: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    if not video_image_dir.exists():
        return idx
    with os.scandir(video_image_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            p = Path(entry.path)
            if p.suffix.lower() in IMAGE_EXTS:
                idx[p.stem] = p
    return idx


def _atomic_json_write(path: Path, payload: dict, *, indent: Optional[int] = None) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding='utf-8')
    tmp.replace(path)


def _format_hms(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return '--:--'
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'


def _stage_image_atomic(src: Path, dst: Path) -> tuple[str, int]:
    ensure_dir(dst.parent)
    src_size = src.stat().st_size
    if dst.exists() and dst.stat().st_size == src_size:
        return 'reused', src_size
    if dst.exists():
        dst.unlink()
    tmp = dst.with_suffix(dst.suffix + '.part')
    if tmp.exists():
        tmp.unlink()
    same_device = False
    try:
        same_device = src.stat().st_dev == dst.parent.stat().st_dev
    except OSError:
        pass
    try:
        if same_device:
            os.link(src, tmp)
        else:
            shutil.copyfile(src, tmp)
    except OSError:
        if tmp.exists():
            tmp.unlink()
        shutil.copyfile(src, tmp)
    got = tmp.stat().st_size
    if got != src_size:
        tmp.unlink(missing_ok=True)
        raise IOError(f'Incomplete stage for {src}: expected {src_size} bytes, got {got}')
    tmp.replace(dst)
    return 'copied', src_size


def _video_dimensions(index: dict[str, Path]) -> tuple[Optional[tuple[int, int]], bool]:
    paths = list(index.values())
    if not paths:
        return None, False
    sample_ids = sorted(set([0, len(paths)//2, len(paths)-1]))
    dims = []
    for i in sample_ids:
        with Image.open(paths[i]) as im:
            dims.append(tuple(map(int, im.size)))
    if len(set(dims)) == 1:
        return dims[0], True
    return None, False


def _fragment_is_complete(fragment: dict, split_dir: Path) -> bool:
    try:
        images = fragment['images']
        if not images:
            return False
        for img in images:
            p = split_dir / img['file_name']
            expected = int(img.get('source_size_bytes', -1))
            if not p.exists():
                return False
            if expected >= 0 and p.stat().st_size != expected:
                return False
        return True
    except Exception:
        return False


def _mirror_prep_status(status_mirror_dir: Optional[Path], payload: dict) -> None:
    if status_mirror_dir is None:
        return
    try:
        _atomic_json_write(status_mirror_dir / 'prep_status.json', payload, indent=2)
    except Exception as exc:
        print(f'⚠️  Não foi possível atualizar prep_status.json no Drive: {exc}', flush=True)


def _build_video_fragment(*, split_name: str, source_split: str, video: str, images_root: Path,
                          labels_root: Path, output_root: Path, progress_every: int,
                          heartbeat_seconds: float, status_mirror_dir: Optional[Path]) -> dict:
    split_dir = ensure_dir(output_root / split_name)
    fragment_dir = ensure_dir(output_root / '.prep_fragments')
    safe_video = video.replace('.', '_')
    fragment_path = fragment_dir / f'{split_name}__{safe_video}.json'
    if fragment_path.exists():
        try:
            fragment = json.loads(fragment_path.read_text(encoding='utf-8'))
            if _fragment_is_complete(fragment, split_dir):
                summary = fragment['summary']
                print(f"♻️  [{split_name.upper()}] {video}: retomado do checkpoint | imagens={summary['images']} | instâncias={summary['instances']} | copiados={summary.get('copied', 0)} | reutilizados={summary.get('reused', 0)}", flush=True)
                return fragment
            print(f'⚠️  [{split_name.upper()}] {video}: checkpoint incompleto; reconstruindo.', flush=True)
        except Exception as exc:
            print(f'⚠️  [{split_name.upper()}] {video}: checkpoint inválido ({exc}); reconstruindo.', flush=True)

    img_dir = images_root / source_split / video
    lbl_dir = labels_root / source_split / video
    if not lbl_dir.exists():
        raise FileNotFoundError(f'Missing segmentation labels: {lbl_dir}')
    if not img_dir.exists():
        raise FileNotFoundError(f'Missing source image folder: {img_dir}')
    print(f'\n🔎 [{split_name.upper()}] {video}: indexando imagens...', flush=True)
    idx = _image_index(img_dir)
    if not idx:
        raise FileNotFoundError(f'No source images found: {img_dir}')
    labels = sorted(lbl_dir.glob('*.txt'), key=lambda p: p.stem)
    if not labels:
        raise FileNotFoundError(f'No segmentation labels found: {lbl_dir}')
    dims, constant_dims = _video_dimensions(idx)
    dims_text = f'{dims[0]}x{dims[1]} constante' if constant_dims and dims else 'resolução variável'
    print(f'▶️  [{split_name.upper()}] {video}: início | labels={len(labels)} | imagens-fonte={len(idx)} | {dims_text}', flush=True)

    images_json: list[dict] = []
    annotations: list[dict] = []
    missing_images: list[str] = []
    local_image_id = 1
    local_ann_id = 1
    copied = reused = copied_bytes = 0
    t0 = time.monotonic(); last_print = t0; last_mirror = 0.0; total = len(labels)
    for n, label_path in enumerate(labels, 1):
        src_img = idx.get(label_path.stem)
        if src_img is None:
            missing_images.append(str(label_path.relative_to(labels_root))); continue
        out_name = f'{safe_video}__{src_img.name}'
        out_img = split_dir / out_name
        action, nbytes = _stage_image_atomic(src_img, out_img)
        if action == 'copied': copied += 1; copied_bytes += nbytes
        else: reused += 1
        if constant_dims and dims: width, height = dims
        else:
            with Image.open(out_img) as im: width, height = map(int, im.size)
        instances = parse_yolo_polygon_label(label_path, width, height)
        images_json.append({'id': local_image_id, 'file_name': out_name, 'width': width, 'height': height,
                            'video_id': video, 'frame_stem': label_path.stem, 'source_split': source_split,
                            'source_size_bytes': int(nbytes)})
        for inst in instances:
            annotations.append({'id': local_ann_id, 'image_id': local_image_id, 'category_id': 1,
                                'segmentation': inst['segmentation'], 'bbox': inst['bbox'],
                                'area': inst['area'], 'iscrowd': 0}); local_ann_id += 1
        local_image_id += 1
        now = time.monotonic()
        should_print = n == 1 or n == total or n % max(1, progress_every) == 0 or (now-last_print) >= heartbeat_seconds
        if should_print:
            elapsed=max(now-t0,1e-6); rate=n/elapsed; eta=(total-n)/rate if rate>0 else None; pct=100*n/total
            print(f'   [{split_name.upper()} {video}] {n:5d}/{total:<5d} ({pct:5.1f}%) | copy={copied} reuse={reused} | inst={len(annotations)} | {copied_bytes/1024**2:8.1f} MiB novos | {rate:5.2f} img/s | ETA {_format_hms(eta)}', flush=True)
            last_print=now
            if status_mirror_dir is not None and (now-last_mirror >= 30.0 or n == total):
                _mirror_prep_status(status_mirror_dir, {'version':'VACA_V1_4_2_PREP_ROBUST','state':'RUNNING','split':split_name,'video':video,'processed':n,'total':total,'percent':round(pct,2),'copied':copied,'reused':reused,'instances_so_far':len(annotations),'rate_images_per_sec':round(rate,3),'eta_sec':round(eta,1) if eta is not None else None,'updated_unix':time.time()}); last_mirror=now
    summary={'images':len(images_json),'instances':len(annotations),'missing_images':len(missing_images),'copied':copied,'reused':reused,'copied_bytes':copied_bytes,'elapsed_sec':round(time.monotonic()-t0,3)}
    fragment={'version':'VACA_V1_4_2_PREP_ROBUST','split':split_name,'source_split':source_split,'video':video,'images':images_json,'annotations':annotations,'missing_images':missing_images,'summary':summary}
    _atomic_json_write(fragment_path,fragment)
    print(f"✅ [{split_name.upper()}] {video}: concluído | imagens={summary['images']} | instâncias={summary['instances']} | copiados={copied} | reutilizados={reused} | tempo={_format_hms(summary['elapsed_sec'])}", flush=True)
    _mirror_prep_status(status_mirror_dir, {'version':'VACA_V1_4_2_PREP_ROBUST','state':'VIDEO_DONE','split':split_name,'video':video,**summary,'updated_unix':time.time()})
    return fragment


def _merge_video_fragments(split_name: str, source_split: str, videos: Iterable[str], fragments: list[dict], output_root: Path) -> dict:
    split_dir=ensure_dir(output_root/split_name); images_json=[]; annotations=[]; missing_images=[]; per_video={}; next_img_id=next_ann_id=1
    for video, fragment in zip(videos,fragments):
        id_map={}
        for img in fragment['images']:
            old=int(img['id']); new_img=dict(img); new_img['id']=next_img_id; new_img.pop('source_size_bytes',None); id_map[old]=next_img_id; images_json.append(new_img); next_img_id+=1
        for ann in fragment['annotations']:
            new_ann=dict(ann); new_ann['id']=next_ann_id; new_ann['image_id']=id_map[int(ann['image_id'])]; annotations.append(new_ann); next_ann_id+=1
        missing_images.extend(fragment.get('missing_images',[])); per_video[video]=dict(fragment['summary'])
    coco={'info':{'description':'VACA V1.4.2 CattleEyeView true COCO instance segmentation','contract':'one source YOLO polygon line = one independent cow instance; no mask union','split':split_name,'prep_version':'VACA_V1_4_2_PREP_ROBUST'},'licenses':[],'categories':[{'id':1,'name':'cow','supercategory':'animal'}],'images':images_json,'annotations':annotations}
    ann_path=split_dir/'_annotations.coco.json'; _atomic_json_write(ann_path,coco)
    return {'split':split_name,'source_split':source_split,'videos':list(videos),'images':len(images_json),'instances':len(annotations),'missing_images':missing_images,'per_video':per_video,'annotation_json':str(ann_path)}


def _build_split(*, split_name: str, source_split: str, videos: Iterable[str], images_root: Path, labels_root: Path, output_root: Path, progress_every: int=25, heartbeat_seconds: float=5.0, status_mirror_dir: Optional[Path]=None) -> dict:
    videos=tuple(videos); print('\n'+'='*78,flush=True); print(f'📦 PREPARANDO SPLIT {split_name.upper()} | vídeos={", ".join(videos)}',flush=True); print('='*78,flush=True)
    fragments=[_build_video_fragment(split_name=split_name,source_split=source_split,video=video,images_root=images_root,labels_root=labels_root,output_root=output_root,progress_every=progress_every,heartbeat_seconds=heartbeat_seconds,status_mirror_dir=status_mirror_dir) for video in videos]
    result=_merge_video_fragments(split_name,source_split,videos,fragments,output_root)
    print(f"🏁 SPLIT {split_name.upper()} pronto | imagens={result['images']} | instâncias={result['instances']} | missing={len(result['missing_images'])}",flush=True)
    _mirror_prep_status(status_mirror_dir, {'version':'VACA_V1_4_2_PREP_ROBUST','state':'SPLIT_DONE','split':split_name,'images':result['images'],'instances':result['instances'],'missing_images':len(result['missing_images']),'updated_unix':time.time()})
    return result


def build_v14_coco_dataset(images_root: Path | str, labels_root: Path | str, output_root: Path | str, *, progress_every: int=25, heartbeat_seconds: float=5.0, status_mirror_dir: Optional[Path | str]=None, force_rebuild: bool=False) -> dict:
    images_root=Path(images_root); labels_root=Path(labels_root); output_root=Path(output_root); status_mirror=Path(status_mirror_dir) if status_mirror_dir is not None else None; manifest_path=output_root/'VACA_V1_4_dataset_manifest.json'
    if force_rebuild and output_root.exists(): print(f'🧹 FORCE_REBUILD: removendo {output_root}',flush=True); shutil.rmtree(output_root)
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8')); print(f"♻️  Dataset COCO já concluído: train={manifest['train']['images']} | valid={manifest['valid']['images']} | test={manifest['test']['images']}",flush=True); _mirror_prep_status(status_mirror, {'version':'VACA_V1_4_2_PREP_ROBUST','state':'DONE','message':'manifest already existed','updated_unix':time.time()}); return manifest
    ensure_dir(output_root)
    if status_mirror is not None: ensure_dir(status_mirror)
    print('🚀 VACA V1.4.2 — preparação COCO-instance robusta',flush=True); print(f'   origem imagens: {images_root}',flush=True); print(f'   origem labels : {labels_root}',flush=True); print(f'   destino local : {output_root}',flush=True); print('   retomada      : por vídeo + reuso de imagens completas',flush=True); print('   monitor Drive : '+(str(status_mirror/'prep_status.json') if status_mirror else 'desativado'),flush=True)
    manifest={'version':'VACA_V1_4_2_PREP_ROBUST','protocol':'video-level holdout; no random-frame primary split','train':_build_split(split_name='train',source_split='train',videos=OFFICIAL_TRAIN_VIDEOS,images_root=images_root,labels_root=labels_root,output_root=output_root,progress_every=progress_every,heartbeat_seconds=heartbeat_seconds,status_mirror_dir=status_mirror),'valid':_build_split(split_name='valid',source_split='val',videos=VACA_VALID_VIDEOS,images_root=images_root,labels_root=labels_root,output_root=output_root,progress_every=progress_every,heartbeat_seconds=heartbeat_seconds,status_mirror_dir=status_mirror),'test':_build_split(split_name='test',source_split='val',videos=VACA_TEST_VIDEOS,images_root=images_root,labels_root=labels_root,output_root=output_root,progress_every=progress_every,heartbeat_seconds=heartbeat_seconds,status_mirror_dir=status_mirror)}
    all_sets=[set(manifest[s]['videos']) for s in ('train','valid','test')]; assert not(all_sets[0]&all_sets[1] or all_sets[0]&all_sets[2] or all_sets[1]&all_sets[2]); missing=sum(len(manifest[s]['missing_images']) for s in ('train','valid','test')); manifest['integrity']={'video_sets_disjoint':True,'missing_image_count':missing,'one_polygon_per_instance':True,'prep_version':'VACA_V1_4_2_PREP_ROBUST'}; _atomic_json_write(manifest_path,manifest,indent=2); _mirror_prep_status(status_mirror, {'version':'VACA_V1_4_2_PREP_ROBUST','state':'DONE','train_images':manifest['train']['images'],'valid_images':manifest['valid']['images'],'test_images':manifest['test']['images'],'total_instances':sum(manifest[s]['instances'] for s in ('train','valid','test')),'missing_image_count':missing,'updated_unix':time.time()}); print('\n✅ PREPARAÇÃO COCO-INSTANCE FINALIZADA',flush=True); print(f"   train={manifest['train']['images']} imagens / {manifest['train']['instances']} instâncias | valid={manifest['valid']['images']} / {manifest['valid']['instances']} | test={manifest['test']['images']} / {manifest['test']['instances']}",flush=True); return manifest


def validate_coco_dataset(dataset_root: Path | str) -> dict:
    from pycocotools.coco import COCO
    dataset_root = Path(dataset_root)
    out = {}
    for split in ("train", "valid", "test"):
        ann = dataset_root / split / "_annotations.coco.json"
        coco = COCO(str(ann))
        image_ids = coco.getImgIds()
        ann_ids = coco.getAnnIds()
        missing = []
        for img in coco.loadImgs(image_ids):
            if not (dataset_root / split / img["file_name"]).exists():
                missing.append(img["file_name"])
        out[split] = {"images": len(image_ids), "instances": len(ann_ids), "missing": missing}
        if missing:
            raise RuntimeError(f"{split}: {len(missing)} images referenced by COCO are missing")
    return out


def draw_random_coco_samples(dataset_root: Path | str, split: str, output_dir: Path | str, n: int = 6, seed: int = 2026) -> list[Path]:
    from pycocotools.coco import COCO
    rng = np.random.default_rng(seed)
    dataset_root, output_dir = Path(dataset_root), ensure_dir(output_dir)
    coco = COCO(str(dataset_root / split / "_annotations.coco.json"))
    ids = coco.getImgIds()
    if not ids:
        return []
    picks = rng.choice(ids, size=min(n, len(ids)), replace=False)
    outputs = []
    for img_id in picks:
        info = coco.loadImgs([int(img_id)])[0]
        frame = cv2.imread(str(dataset_root / split / info["file_name"]))
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[int(img_id)]))
        overlay = frame.copy()
        for j, ann in enumerate(anns, 1):
            mask = coco.annToMask(ann).astype(bool)
            color = track_color(j)
            overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array(color)).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
        cv2.putText(overlay, f"{split} | {info['file_name']} | instances={len(anns)}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        out = output_dir / f"sample_{split}_{img_id}.jpg"
        cv2.imwrite(str(out), overlay)
        outputs.append(out)
    return outputs


# -----------------------------
# RF-DETR
# -----------------------------
def gpu_training_profile() -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for VACA V1.4 RF-DETR training")
    props = torch.cuda.get_device_properties(0)
    mem_gb = props.total_memory / (1024**3)
    if mem_gb < 10:
        profile = {"batch_size": 1, "grad_accum_steps": 16, "gradient_checkpointing": True}
    elif mem_gb < 18:
        profile = {"batch_size": 2, "grad_accum_steps": 8, "gradient_checkpointing": True}
    elif mem_gb < 30:
        profile = {"batch_size": 3, "grad_accum_steps": 4, "gradient_checkpointing": False}
    else:
        profile = {"batch_size": 4, "grad_accum_steps": 4, "gradient_checkpointing": False}
    profile.update({"gpu_name": props.name, "memory_gb": round(mem_gb, 2)})
    return profile


def find_best_rfdetr_checkpoint(output_dir: Path | str) -> Optional[Path]:
    output_dir = Path(output_dir)
    preferred = [output_dir / "checkpoint_best_total.pth", output_dir / "checkpoint_best_regular.pth", output_dir / "checkpoint_best_ema.pth"]
    for p in preferred:
        if p.exists(): return p
    candidates = sorted(list(output_dir.glob("*.pth")) + list(output_dir.glob("*.ckpt")), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def find_resume_checkpoint(output_dir: Path | str) -> Optional[Path]:
    output_dir = Path(output_dir)
    for name in ("last.ckpt", "checkpoint_last.pth"):
        p = output_dir / name
        if p.exists(): return p
    ckpts = sorted(output_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


def train_rfdetr_seg_medium(dataset_root: Path | str, output_dir: Path | str, *, epochs: int = 100, lr: float = 1e-4, resume: bool = True, early_stopping_patience: int = 15) -> tuple[object, dict]:
    from rfdetr import RFDETRSegMedium
    dataset_root=Path(dataset_root); output_dir=ensure_dir(output_dir)
    if not (dataset_root/'train'/'_annotations.coco.json').exists(): raise FileNotFoundError(f'RF-DETR COCO train JSON missing: {dataset_root}')
    if not (dataset_root/'valid'/'_annotations.coco.json').exists(): raise FileNotFoundError(f'RF-DETR COCO valid JSON missing: {dataset_root}')
    profile=gpu_training_profile(); print('\n'+'='*78,flush=True); print('🧠 RF-DETR SEG MEDIUM — TREINO VACA',flush=True); print('='*78,flush=True); print(f"GPU: {profile['gpu_name']} | VRAM={profile['memory_gb']} GiB",flush=True); print(f"batch={profile['batch_size']} | grad_accum={profile['grad_accum_steps']} | gradient_checkpointing={profile['gradient_checkpointing']} | resolution=432",flush=True); print(f'dataset={dataset_root}',flush=True); print(f'checkpoints/logs={output_dir}',flush=True)
    model=RFDETRSegMedium(gradient_checkpointing=profile['gradient_checkpointing'])
    kwargs=dict(dataset_dir=str(dataset_root),epochs=int(epochs),batch_size=int(profile['batch_size']),eval_batch_size=1,grad_accum_steps=int(profile['grad_accum_steps']),lr=float(lr),output_dir=str(output_dir),resolution=432,device='cuda',amp_dtype='auto',early_stopping=True,early_stopping_patience=int(early_stopping_patience),early_stopping_min_delta=0.001,skip_best_epochs=3,checkpoint_interval=5,scale_jitter=False,augmentation_backend='torchvision',tensorboard=True,num_workers=2)
    resume_path=find_resume_checkpoint(output_dir) if resume else None
    if resume_path is not None: kwargs['resume']=str(resume_path); print(f'♻️  Retomando treinamento exato de: {resume_path}',flush=True)
    else: print('▶️  Iniciando fine-tuning a partir dos pesos COCO pré-treinados.',flush=True)
    model.train(**kwargs); profile['resume_checkpoint']=str(resume_path) if resume_path else None; profile['train_api_profile']='RFDETR_1_10_COLAB_SAFE'; return model, profile

# Remaining VACA tracking/Cutie/renderer functions are retained in the Colab-embedded helper.
# This GitHub file records the PREP-ROBUST and RF-DETR training portions used by V1.4.2.
