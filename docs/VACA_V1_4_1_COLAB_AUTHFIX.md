# VACA V1.4.1 — Colab Drive AUTHFIX

Date: 2026-09-15

## Problem observed

Google Colab raised:

`MessageError: Error: credential propagation was unsuccessful`

inside `google.colab.auth.authenticate_user()` before VACA could create a Drive API client.

This is an upstream Colab authentication failure, not an RF-DETR/Cutie failure.

## Fix

VACA V1.4.1 no longer calls the Drive API on the normal path when the dataset is already exposed by the mounted Google Drive shortcut.

Primary dataset root checked first:

`/content/drive/.shortcut-targets-by-id/1Pjt1bEZR5s-grzMqnuOEpIv9A7JK1VoL/dataset`

From that root VACA reads directly:

- `annotation/detect/images`
- `annotation/segment/labels`
- `videos`

`auth.authenticate_user()` is now fallback-only and is invoked only if mounted data cannot be found.

The video acquisition cell was also changed to copy each requested MP4 from the mounted Drive shortcut to local Colab SSD before inference. This both avoids the auth bug and reduces Drive I/O during GPU processing.

## Official patched artifacts

Saved in Drive:

- `VACA_DATA/12_DOCUMENTATION_EXPORTS/VACA_V1_4_1_MULTI_COW_RFDETR_CUTIE_COLAB_AUTHFIX.ipynb`
- `VACA_DATA/12_DOCUMENTATION_EXPORTS/vaca_v14_colab_pipeline_authfix.py`

Notebook SHA-256:

`4bd3ab53de7a8cd75f532be8b722537bc7da58e2cb52e9ccacd739d226363e6f`

Helper SHA-256:

`36dde6839993fc0eafb39752dc93377d4cf94bf2a421d2d28b687bb0ef81b987`

Validation performed before publication:

- notebook JSON valid;
- 21 total cells;
- 16 Python code cells;
- all code cells compile;
- helper passes `py_compile`.
