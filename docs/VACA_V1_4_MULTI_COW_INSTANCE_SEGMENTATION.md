# VACA V1.4 — Multi-Cow Instance Segmentation & Identity-Stable Temporal Fusion

## Decision

V1.4 replaces the semantic/foreground GROUP logic as the primary perception mechanism. The production problem is defined as **visible instance segmentation + temporal identity continuity**.

Touching cows must remain independent instances whenever evidence supports separation. When evidence is insufficient (especially complete occlusion), VACA preserves track state and emits REVIEW rather than hallucinating an individual mask or Cow ID.

## Why V1.3 cannot be repaired by thresholds

V1.3 can isolate foreground but merges touching cattle. A center detector/watershed style splitter is not promoted because it can invent boundaries when animals overlap. The correct representation is one annotation polygon = one cow instance, followed by a temporal model that preserves instance identity across contact.

## SOTA audit — September 2026

### Production detector/segmenter candidates

1. **RF-DETR Seg Small/Medium/Large — primary challenger**
   - End-to-end transformer instance segmentation with current Hugging Face/Roboflow checkpoints.
   - Production-safe core sizes are currently documented under Apache-2.0; verify the exact selected checkpoint again at freeze time.
   - Strong modern accuracy/latency design with a DINOv2-style backbone.
   - Known caveat: overlapping predicted masks can occur under partial occlusion, so raw RF-DETR masks are not accepted as final visible ownership.

2. **RTMDet-Ins-M — mandatory incumbent benchmark**
   - Mature OpenMMLab implementation and ONNX/TensorRT deployment path.
   - Apache-2.0 ecosystem.
   - Official COCO mask AP: 42.1 for RTMDet-Ins-M.
   - Production advantage: mature deployment and predictable latency.

3. **MaskDINO R50/Swin-L — offline hard-case teacher/challenger**
   - Apache-2.0.
   - Official COCO Mask AP about 46.3 (R50) and 52.3 (Swin-L).
   - Too heavy to assume as the edge default, but useful as an offline adjudicator/teacher on difficult touching-cow frames.

### Temporal segmentation

**Cutie** is selected as the production temporal mask-memory engine:
- CVPR 2024 Highlight; object-level + pixel-level memory.
- MIT license.
- Multiple objects and explicit add/remove object workflows.
- Published multi-animal work used bidirectional Cutie inference and disagreement zones to strongly reduce identity swaps during close interactions.

**SAM 2.1** remains a repair/challenger tool, not the sole identity tracker. It supports multi-object video segmentation with streaming memory and independent per-object inference, but public issues document missing multi-object masks and identity jumping between visually similar objects. VACA therefore never lets SAM2 alone decide permanent track identity.

**DEVA** provides the architectural precedent for decoupling image-level segmentation from universal temporal propagation and fusing both sources over time. VACA adopts this principle with cattle-domain instance masks.

## Final V1.4 perception architecture

```text
frame t
  |
  +--> RF-DETR-Seg-M (primary) -------+
  |                                    |
  +--> RTMDet-Ins-M (benchmark)        | spatial instance evidence
  |                                    |
  +--> MaskDINO (offline hard cases) --+
                                       |
                              Instance Candidate Set
                                       |
                                       v
                    Visible-Pixel Ownership Resolver
                    (one visible pixel <= one cow)
                                       |
                +----------------------+----------------------+
                |                                             |
                v                                             v
        Cutie forward memory                         Detector re-anchor
        per passage track                            high-confidence masks
                |
       cow leaves passage
                |
                v
        Cutie reverse pass
                |
                v
      Forward/Reverse Consensus
                |
                v
        Identity Governor
    motion + mask + area + axis
    + appearance/Re-ID embedding
                |
    +-----------+-------------+
    |                         |
    v                         v
CONFIRMED INSTANCE       CONFLICT / OCCLUSION
track_id                 REVIEW / OCCLUDED
    |                         |
    +------------+------------+
                 v
         RTMPose Bovine15
       (only confirmed masks)
```

## Hard invariants

1. **One YOLO polygon line = one independent cow instance. Never union training masks.**
2. A visible pixel cannot be assigned to two cows in the final visible-instance map.
3. Detector and temporal propagator are independent evidence sources; one cannot silently overwrite the other.
4. `track_id` is passage-local and is never a permanent `cow_id`.
5. During unresolved contact/occlusion, permanent Cow ID is not promoted.
6. A complete occlusion is represented as `OCCLUDED`, not an invented full-body mask.
7. Reappearance must pass reacquisition gates before returning to `VISIBLE`.
8. Pose and mobility models consume only CONFIRMED masks/tracks. REVIEW frames are excluded or down-weighted.
9. Every production checkpoint is pinned by repository commit + model checksum + dataset manifest.

## Identity Governor

For each current spatial instance vs each active temporal track, compute a gated association cost using:
- visible-mask IoU / boundary overlap;
- centroid displacement relative to constant-velocity prediction;
- area ratio;
- longitudinal-axis/orientation consistency;
- coat-pattern appearance embedding / later open-set Re-ID embedding;
- agreement with Cutie propagated mask;
- detector confidence and temporal confidence.

State machine:

```text
TENTATIVE -> VISIBLE -> OCCLUDED -> REACQUIRE -> VISIBLE
                    \                         /
                     -> REVIEW --------------
                     -> LOST
```

### Pixel ownership conflict

When two predicted masks overlap:
- if one hypothesis has clearly stronger joint spatial+temporal evidence, that visible pixel is assigned to it;
- if the evidence margin is insufficient, the pixel is unowned in the finalized map and both tracks are flagged REVIEW;
- VACA does **not** preserve overlapping final visible masks merely because a detector produced them.

## Bidirectional passage finalization

The corridor setup allows a strong reliability mechanism:
1. Online: detector + forward Cutie gives provisional live tracks.
2. When a passage finishes, choose a high-confidence final anchor and run Cutie backward.
3. Compare forward and backward masks for each track.
4. High agreement -> finalize visible masks.
5. Local disagreement -> Zone of Disagreement / REVIEW; optionally request MaskDINO or SAM2.1 repair.

This lets a short cattle passage be finalized after the animal exits rather than making irreversible online identity decisions during contact.

## Training data protocol

### CattleEyeView
- Convert segmentation labels directly to COCO instance segmentation.
- Preserve every line as a separate instance.
- Preserve frame/video/camera metadata.
- No random-frame-only primary split.

### VACA farm data
Required before production promotion:
- touching cows;
- partial occlusions;
- full temporary occlusions;
- entry/exit at image border;
- black/white coat extremes;
- wet/dirty animals;
- lighting changes;
- multiple camera heights/lenses/farms.

Hard examples may be oversampled in training, but test sets remain frozen and untouched.

## Evaluation

Semantic IoU from V1.3 is no longer the primary promotion metric. V1.4 must report:
- COCO mask AP, AP50, AP75;
- per-instance recall;
- merge error rate (one predicted visible instance consumes >1 GT cow);
- split error rate (one GT cow becomes multiple predictions);
- visible-mask overlap rate after ownership resolver (must be zero by contract);
- HOTA / AssA / DetA where available;
- IDF1 and ID switches;
- occlusion recovery accuracy;
- time-to-reacquire;
- fraction of frames/instances sent to REVIEW;
- latency, VRAM/RAM and throughput.

### Proposed engineering promotion gates (VACA targets, not literature claims)
- mask AP50 >= 0.90 on in-domain video holdout;
- mask AP75 >= 0.75;
- merge error <= 2% of multi-cow frames;
- IDF1 >= 0.95 for passage tracks;
- <= 1 ID switch / 1000 track-frames, with target zero on video 01;
- 100% zero-overlap in finalized visible ownership maps;
- zero silent Cow-ID promotion during unresolved overlap;
- external camera/farm holdout before production.

Failure of any identity/merge gate keeps the model experimental regardless of visual quality.

## Benchmark order

1. Build immutable COCO-instance manifests from CattleEyeView polygons.
2. Train/fine-tune RF-DETR-Seg-M and RTMDet-Ins-M on identical splits/augmentations.
3. Run MaskDINO R50 as offline hard-case challenger.
4. Select the spatial winner by instance metrics, not semantic IoU.
5. Add Cutie seeded/reseeded with winner masks.
6. Add forward/reverse passage consensus.
7. Add Identity Governor + coat-pattern embedding.
8. Evaluate complete video 01, then all 14 videos, then external farm/camera holdout.
9. Only then feed confirmed tracks into RTMPose-Bovine15.

## Licensing guardrail

Production core must remain compatible with commercial deployment:
- RF-DETR core sizes: verify exact selected checkpoint/model license at freeze time; current official core docs indicate Apache-2.0.
- RTMDet/MMDetection: Apache-2.0.
- Cutie: MIT.
- MaskDINO: Apache-2.0.
- SAM2: Apache-2.0.
- Annolid and SAM2Long contain non-commercial licensing constraints and are research/design references only unless separately cleared; do not copy them into the production core.

## Definition of 'definitive solution'

No vision system can recover the true invisible contour/identity from an arbitrarily long complete occlusion with zero distinguishing evidence. VACA's definitive behavior is therefore: **separate every supported visible instance, preserve identity through temporal memory, reacquire using multiple cues, and explicitly REVIEW fundamentally ambiguous events rather than hallucinating.**
