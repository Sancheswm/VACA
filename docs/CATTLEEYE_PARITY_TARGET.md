# VACA — CattleEye parity target

## Product ambition

VACA is not defined as a single lameness classifier. The target is a production-grade, continuous cattle monitoring platform with functional parity to the public capabilities of CattleEye, while using an independently developed architecture and validation process.

## Public benchmark capabilities to match

1. Overhead 2D camera installation in a cow traffic corridor, preferably at the milking parlour exit.
2. Automatic individual cow identification without relying on a permanent wearable as the primary inference input.
3. Per-pass mobility analysis and longitudinal mobility score.
4. Automated body condition scoring (BCS).
5. Daily individual history and herd-level trends.
6. Change detection and prioritised management alerts.
7. Edge-capable processing with local continuity when internet connectivity is unavailable.
8. Cloud synchronisation and web/mobile dashboard.
9. Integration with herd-management identifiers and systems where available.
10. Auditable model outputs and clinically conservative labelling.

## VACA target architecture

video capture
→ capture-quality audit
→ corridor calibration
→ cattle detection / segmentation
→ multi-object tracking
→ individual re-identification
→ Bovine15 pose estimation
→ gait feature extraction
→ temporal representation model
→ mobility score + uncertainty
→ body condition model
→ longitudinal cow profile
→ alert/ranking engine
→ edge database / sync
→ API
→ dashboard

## Required model families

### Perception
- cattle detector / instance segmenter
- multi-object tracker
- Bovine15 pose estimator
- visual re-identification model

### Mobility
The production model must not depend on one handcrafted dorsal angle. It must learn from a multivariate temporal representation including, where measurable:
- dorsal trajectory and lateral sway
- stride timing and symmetry
- limb phase relationships
- walking velocity and acceleration
- path curvature
- body orientation
- head/neck motion
- hip/shoulder motion asymmetry
- pose confidence and occlusion information

Candidate temporal families include ST-GCN++, CTR-GCN, temporal convolution, transformer-based skeleton/video models and calibrated ensembles. Selection must be empirical.

### Body condition
BCS requires its own independently validated model, based on dorsal body geometry/appearance and longitudinal consistency. Mobility and BCS must not be conflated into one output.

### Identity
Cow identification must have explicit metrics for:
- identification coverage
- identification accuracy
- time-to-enrolment
- unknown/new-animal handling
- duplicate identity detection

## Validation gates

A model cannot be presented as clinically useful solely because it looks plausible on video.

### Gate 1 — perception
Validate detection, segmentation, tracking and pose on held-out animals and recordings.

### Gate 2 — identity
Validate cow ID independently from mobility and BCS.

### Gate 3 — mobility agreement
Compare automated mobility outputs against multiple trained human/veterinary scorers and report agreement, discrimination and calibration.

### Gate 4 — lesion association
Where hoof-trimming/lesion records exist, test whether the score predicts clinically relevant painful lesions.

### Gate 5 — longitudinal validation
Use farm-level and animal-level holdouts to prevent leakage and determine whether temporal change improves early warning.

### Gate 6 — prospective field validation
Run prospectively on a commercial herd without changing labels after seeing predictions.

### Gate 7 — intervention study
The long-term gold standard is a prospective intervention design evaluating whether use of VACA alerts improves animal outcomes compared with standard management.

## Initial acceptance targets

These are engineering targets, not current performance claims.

- cattle detection/segmentation: production-quality performance on the target camera geometry
- tracking: stable identity through the complete corridor transit
- visual cow ID: ≥98% precision among confidently identified animals, with explicit unknown rejection
- mobility: agreement with trained assessors within the range reported for inter-human scoring, validated on multiple farms
- BCS: error/agreement comparable with trained assessors on the intended scoring scale
- uptime: continuous local operation during internet outage and later cloud synchronisation
- explainability: every alert links to score history, confidence and supporting video/trajectory evidence

## Competitive differentiation goals

VACA should aim not only for parity, but for advantages relevant to Brazilian dairy systems:
- transparent uncertainty and evidence traces
- explicit Bovine15 biomechanics
- modular open research architecture
- local/edge-first operation
- lower-cost deployment with commodity cameras
- Portuguese-first interface
- configurable integration with Brazilian herd-management workflows
- research-grade experiment and dataset provenance

## Scientific guardrail

Until expert-reviewed ground truth and prospective validation exist, outputs must be labelled as research/technical scores rather than veterinary diagnoses.
