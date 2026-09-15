# VACA V1.4.2 — PREP-ROBUST

## Purpose

This patch fixes the silent CattleEyeView COCO-instance preparation bottleneck observed in Google Colab when staging thousands of images from the mounted Drive shortcut into `/content`.

## Changes

- direct mounted-Drive workflow remains primary; Google Drive API auth is fallback only;
- Drive/FUSE -> `/content` no longer attempts a failing hardlink for every image;
- staging uses `.part` + atomic rename and byte-size validation;
- live progress prints include split, video, images processed, percent, copied/reused count, instance count, MiB copied, images/s and ETA;
- a per-video fragment checkpoint is written under `.prep_fragments`;
- interrupted runs reuse already-complete staged images;
- completed video fragments are loaded directly on rerun;
- `prep_status.json` is mirrored to `VACA_DATA/00_MANIFESTS/<RUN_TAG>/` about every 30 seconds and on video/split completion;
- final COCO JSON is assembled from video fragments with globally consistent image/annotation IDs;
- training call audited for current RF-DETR 1.10.x semantics: constructor-level gradient checkpointing, full `.ckpt` resume, `scale_jitter=False`, pinned torchvision augmentation backend, 5-epoch checkpoint cadence, and `skip_best_epochs=3`.

## Colab artifact

Drive:

`VACA_DATA/12_DOCUMENTATION_EXPORTS/VACA_V1_4_2_MULTI_COW_RFDETR_CUTIE_PREP_ROBUST.ipynb`

Convenience latest pointer:

`VACA_DATA/12_DOCUMENTATION_EXPORTS/VACA_LATEST_COLAB.ipynb`

## Resume semantics

If preparation is interrupted mid-video, the next run reparses that video's labels but byte-size checks and reuses every complete local image already staged. If a video's fragment checkpoint exists and all referenced staged images remain valid, that video is skipped entirely and loaded from the fragment.

If the Colab runtime itself is destroyed, `/content` is lost and source images must be staged again; model checkpoints and project outputs remain in Drive.

## Verification performed

- helper `py_compile`: PASS
- helper import: PASS
- synthetic train/valid/test preparation covering all 14 video IDs: PASS
- independent COCO instance generation: PASS
- second-run resume: PASS
- live status final state `DONE`: PASS
- notebook code-cell compilation: 16/16 PASS

## Scientific split invariant

Train: 02,03,04,06,08,11,12,13,14

Valid: 05,09

Test: 01,07,10

The three video sets are disjoint. Video 01 remains excluded from training.
