# VACA V1.4 — Google Colab Runbook

## Purpose

This runbook freezes the executable Google Colab entry point for the VACA V1.4 multi-cow instance pipeline. The notebook is an orchestration artifact; scientific architecture, instance-label contract and promotion rules remain versioned in GitHub.

## Frozen Colab artifact

Notebook:
`VACA_V1_4_MULTI_COW_RFDETR_CUTIE_COLAB.ipynb`

Google Drive:
`VACA_DATA/12_DOCUMENTATION_EXPORTS/VACA_V1_4_MULTI_COW_RFDETR_CUTIE_COLAB.ipynb`

Drive file ID:
`1MxQMhK3IWFQN5nui2zZ7xYMIoJ0CFrDv`

SHA256:
`1114794e43e91f917c7d171ef3ded6abfa0ee2b66234c1934af10814ab747ef4`

Embedded/helper source:
`VACA_DATA/12_DOCUMENTATION_EXPORTS/vaca_v14_colab_pipeline.py`

Drive file ID:
`1X7-mjTHNWc8bjdZJT19CDuSp_rc2JylH`

SHA256:
`a2c5186e52fb60a57c323103aed1fdd58b48b02047113c576a4f573a9e231c99`

The notebook embeds the helper bytes, so the notebook is self-contained after download; the separate helper copy exists for audit and diffing.

## Runtime

Use a Google Colab GPU runtime. CUDA is mandatory; the notebook fails early rather than silently running a benchmark on CPU.

Recommended order:
1. Open the frozen `.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Run all cells from top to bottom.
4. Authorize Google Drive when prompted.
5. Do not edit source paths unless the Drive layout has intentionally changed.

## Verified input sources

CattleEyeView dataset root Drive ID:
`1Pjt1bEZR5s-grzMqnuOEpIv9A7JK1VoL`

Videos folder:
`1_vUuM3_5evffsszdOcnEoOoc1H9j2F7Y`

Segmentation labels root:
`1x1gMFYPol99qpL5Dz3bukARXDaQvgfEm`

Detection-image root reused for instance training:
`16COroSFlEt1uFAwJhWlnpmdCK9EeT7KH`

Video 01 file ID:
`1FgAckqvHJKvbw_dpnHMTMD1vVtCLUNga`

Expected video-01 integrity:
- 1920x1080
- 8 fps
- 1226 frames
- 153.25 s
- SHA256 `e80374a4659994b2361afbe04bc042ba8967fd98c2e11ff89611456d34014568`

The notebook first looks for the mounted Drive tree and falls back to Google Drive API folder IDs when shortcuts are not exposed in the Colab mount.

## Leakage-safe split

CattleEyeView's published YOLO organization duplicates test/validation image sets. VACA therefore uses video-level partitions:

Training videos:
`02, 03, 04, 06, 08, 11, 12, 13, 14`

Validation videos:
`05, 09`

Test videos:
`01, 07, 10`

Video 01 is never used for RF-DETR fine-tuning.

## Instance-label invariant

One YOLO polygon line equals one independent cow instance.

The Colab never unions touching-cow polygons into one semantic foreground target. This is the key change from V1.3.

## Default spatial model

Primary model:
`RFDETRSegMedium`

Pinned Python package:
`rfdetr==1.10.1`

Pinned visualization/evaluation package:
`supervision==0.30.2`

Default training parameters:
- 100 epochs
- lr 1e-4
- automatic GPU-aware batch profile
- resume enabled
- early stopping enabled
- gradient checkpointing available/used according to GPU profile
- output checkpoints written directly to Drive

## Temporal model

The notebook clones the official `hkchengrex/Cutie` repository, downloads official weights using the repository utility and records the exact Git commit in the final manifest.

Each passage uses object-memory segmentation with periodic RF-DETR re-anchors. A reverse pass is performed after the forward sequence when bidirectional auditing is enabled.

Forward/reverse disagreement does not silently change identity. It becomes an audit/REVIEW signal.

## Identity and overlap rules

- one visible pixel can have at most one finalized cow owner;
- detector hypotheses are associated with active passage tracks using mask IoU, centroid motion and area consistency;
- unresolved spatial conflicts are not converted into permanent Cow IDs;
- complete/ambiguous occlusion is retained as temporal uncertainty rather than hallucinated anatomy;
- no RTMPose/mobility diagnosis is emitted by this notebook.

A learned coat-pattern Re-ID branch will be added after this spatial/temporal gate is validated; `track_id` in V1.4 is still passage-local.

## Benchmarks emitted

The notebook writes:
- COCO segmentation evaluation;
- per-instance merge/split audit on the test split;
- forward telemetry;
- forward/reverse consensus report;
- model/checkpoint hashes;
- exact package/model/run manifest;
- final annotated H.264 video.

Semantic foreground IoU from V1.3 is no longer a sufficient promotion metric.

## Default output tree

All heavy artifacts remain in Drive:

`VACA_DATA/03_PROCESSED/VACA_V1_4_RFDETR_CUTIE/`

`VACA_DATA/04_CHECKPOINTS/VACA_V1_4_RFDETR_CUTIE/RFDETR_SEG_M/`

`VACA_DATA/05_RUNS/VACA_V1_4_RFDETR_CUTIE/VIDEO_XX/`

`VACA_DATA/06_OUTPUTS/VACA_V1_4_RFDETR_CUTIE/VIDEO_XX/`

`VACA_DATA/08_REPORTS/VACA_V1_4_RFDETR_CUTIE/`

`VACA_DATA/10_TEMP_STAGING/VACA_V1_4_RFDETR_CUTIE/`

`VACA_DATA/14_BENCHMARKS/VACA_V1_4_RFDETR_CUTIE/`

## Default video scope

The frozen notebook starts with:
`VIDEO_IDS_TO_PROCESS = ["01"]`

After video 01 passes the instance/identity audit, switch to all fourteen videos:
`VIDEO_IDS_TO_PROCESS = [f"{i:02d}" for i in range(1, 15)]`

Do not process all videos first and then discover a systematic identity defect.

## Resume/recovery

`RESUME_TRAINING = True` by default. Checkpoints are stored in Drive, so a disconnected Colab session can resume from the latest compatible checkpoint.

Per-video outputs are also written to Drive. Rerunning the notebook does not require downloading the 12.2 GB `images.tar.gz`; it uses the YOLO image subset in `annotation/detect/images` together with `annotation/segment/labels`.

## Promotion rule

The notebook being executable does **not** promote V1.4. Promotion occurs only after the actual GPU run reports acceptable instance AP, merge/split error and temporal identity metrics. Visual quality alone is not a gate.

The first required audit after execution is video 01, especially the dense multi-cow interval that caused V1.3 `GROUP/REVIEW` failures.
