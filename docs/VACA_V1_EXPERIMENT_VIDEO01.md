# VACA V1 Experimental — Video 01

Status: **engineering baseline, not clinically validated**

## Objective

Exercise the VACA V1 pipeline on CattleEyeView `01.mp4` before trained RTMDet, BoT-SORT, Re-ID, RTMPose-Bovine15, temporal mobility and BCS checkpoints are promoted.

This run deliberately does **not** emit a lameness diagnosis or clinical score.

## Input

- Dataset: CattleEyeView
- Video: `01.mp4`
- Drive file ID: `1FgAckqvHJKvbw_dpnHMTMD1vVtCLUNga`
- Native video: 1920×1080, 8 FPS, 1226 frames, 153.25 s
- Segmentation labels: official CattleEyeView validation split, sampled frames from video 01

## Experimental pipeline

```text
01.mp4
  -> capture audit
  -> illumination-compensated motion segmentation [TEMPORARY]
  -> oriented foreground envelope / body-axis proxy
  -> centroid association tracker [TEMPORARY]
  -> passage-group detection
  -> telemetry + validation + annotated video
```

The fallback stages are intentionally isolated behind the same conceptual interfaces that will later be replaced by:

```text
RTMDet-Ins -> BoT-SORT -> open-set ReID -> RTMPose-Bovine15 -> temporal mobility model
```

## Measured result

The experiment detected two high-occupancy passage groups:

1. ~2.9–36.8 s
2. ~113.1–145.3 s

Against five official CattleEyeView segmentation annotations near the end of video 01, the classical fallback achieved:

- mean IoU, all 5 sampled frames: **0.371**
- mean IoU on frames where annotated cattle occupy >=1% of the image: **0.464**
- mean pixel precision: **0.694**
- mean pixel recall: **0.396**

This is **not** an acceptable production segmentation result. It is valuable because it gives VACA a measured, reproducible floor that RTMDet-Ins must beat.

## Interpretation

The experiment proves that the end-to-end engineering path is functioning:

- original Drive video can be ingested;
- capture geometry can be audited;
- foreground passages can be discovered;
- masks and oriented envelopes can be generated;
- provisional tracks can be recorded;
- telemetry and manifests can be produced;
- results can be validated against official dataset annotations.

It does **not** prove individual cow tracking, identity, pose, BCS or lameness detection. Large connected components are explicitly labelled `GROUP`; track IDs are engineering identifiers only and can switch when animals overlap.

## Promotion gate

The next detector experiment must replace the fallback with RTMDet-Ins and beat this baseline on the same validation frames and then on the complete held-out CattleEyeView validation split.

No clinical model is allowed to consume these provisional tracks as ground truth.
