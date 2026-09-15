# VACA V1.4.3 — Video Frame Recovery

## Why this revision exists

The CattleEyeView segmentation labels contain more annotated frame stems than the `annotation/detect/images` tree contains JPEGs. During the V1.4.2 preparation run, the integrity gate reported:

- train: 7,591 staged images, 20,088 instances, 1,283 annotated frames missing a source image
- valid: 890 staged images, 2,295 instances, 159 annotated frames missing a source image
- test: 1,600 staged images, 6,128 instances, 196 annotated frames missing a source image
- total unresolved annotated frames: 1,638

The generated COCO files were internally valid, but training was correctly blocked because `missing_image_count != 0`.

## V1.4.3 policy

V1.4.3 never silently drops an annotated frame and never guesses frame numbering.

For each affected video it:

1. Reuses complete V1.4.2 staged images and complete per-video fragments.
2. Copies the original MP4 to local Colab SSD only when recovery is required.
3. Uses official image/video anchor pairs to calibrate the mapping between filename stem `N` and zero-based video frame index by testing offsets `N-2 ... N+2`.
4. Compares anchors using a composite similarity score, MAE, correlation and PSNR.
5. Recovers missing annotated frames only if calibration has strong absolute similarity and a discriminative margin/majority.
6. Aborts rather than guessing if calibration is ambiguous.
7. Decodes missing targets in a forward pass, writes JPEGs atomically, validates dimensions, and records provenance (`source_kind=video_recovered`, `video_frame_index`, calibrated offset).
8. Rebuilds affected COCO fragments and requires final `missing_image_count == 0` before RF-DETR training.

## Resume and observability

The preparation remains resumable. It preserves:

- staged images under `/content/vaca_v14_work/cattleeye_coco_instance_v14/{train,valid,test}`
- per-video fragments under `.prep_fragments/`
- recovered frames under `.recovered_frames/`
- local video cache under `/content/vaca_v14_work/video_recovery_cache/`
- live Drive status under `VACA_DATA/00_MANIFESTS/<RUN_TAG>/prep_status.json`

Console output includes calibration metrics, frame-recovery progress, staged-image progress, counts of reused/copied/recovered images, instances, ETA, and final integrity.

## Validation performed before publication

A synthetic adversarial test was executed across the same 14-video split topology. Each synthetic video had complete labels but only a subset of official JPEG frames. V1.4.3 correctly calibrated offset `0`, recovered all missing frames, produced `missing_image_count=0`, and a second run reused the completed manifest without repeating work.

The notebook also passed structural validation: 21 cells, 16 Python code cells, zero compile errors, and the embedded helper is byte-identical to the standalone helper.

## Drive artifacts

Official V1.4.3 artifacts are stored in `VACA_DATA/12_DOCUMENTATION_EXPORTS/`:

- `VACA_V1_4_3_MULTI_COW_RFDETR_CUTIE_VIDEO_RECOVERY.ipynb`
- `vaca_v14_colab_pipeline_v143.py`
- `VACA_LATEST_V143_COLAB.ipynb`
- `vaca_v14_colab_pipeline_LATEST_V143.py`

Do not start RF-DETR training until the real CattleEyeView run reports final `missing_image_count = 0` and `pycocotools` validation passes for train/valid/test.
