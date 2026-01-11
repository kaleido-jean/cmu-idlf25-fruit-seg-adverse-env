
# Robust Fruit Segmentation for Automated Harvesting Robots under Adverse Environmental Conditions

Course project (CMU 11-785) on **green pepper detection & segmentation robustness** in **low-light and foggy** settings.

## Methodology Overview
We compared two robustness pipelines:

### (A) Illumination-invariant feature learning (feature-level robustness)
- Insert an **Illumination-Invariant Module (IIM)** (from **YOLA**) before an off-the-shelf detector.
- Goal: learn features less sensitive to illumination changes.

### (B) Image pre-processing (input-level robustness)
- Apply a low-light enhancement / differentiable image processing front-end (tested: **SGZ**, attempted: **IA-YOLO-style**), then run the detector/segmenter.

Both pipelines evaluate on normal-light training plus adverse-condition testing.

## Data
- Training uses an open-source **bell pepper dataset** \[1\]: **2,127 images** with some variation (indoor/outdoor, occlusion, mild blur) but limited adverse-condition coverage.
- Additional adverse-condition images were used for evaluation, including:
  - **PIL-based pixel-level edits** (brightness/contrast/saturation; synthetic rain/fog overlays)
  - **Real outdoor images** provided by a CMU MRSD team (VADER)

## Models
### Baselines
- **YOLOv9c-seg** (fine-tuned from pretrained weights)
- **Detectron2 Mask R-CNN** (`mask_rcnn_R_50_FPN_3x`, COCO-pretrained initialization)

### Robustness Extensions
- **YOLA + YOLOv3** (IIM inserted for illumination-invariance)
- **SGZ** as preprocessing for low-light enhancement
- **IA-YOLO** adaptation attempt (not successfully reproduced due to training/implementation issues)

## Evaluation Metrics
- Primary: **mAP@0.5**, **Precision**, **Recall**
- Also considered: inference efficiency (time/memory/params) as reference

## Key Results
### Baseline reproduction
- Fine-tuned YOLOv9 achieved strong performance on normal condition, with degradation under low-light/fog but remained relatively resilient.

### Illumination-invariant feature learning (YOLA)
- Improved **mAP50** and especially **Recall** under low-light testing, but with a major **Precision drop** (more false positives).
- Interpreted as a “recall booster” that may be useful where missing objects is costly, but may be undesirable for high-precision harvesting unless additional false-positive suppression is added.

### Image pre-processing (SGZ)
- SGZ improved perceived visibility but **hurt detection** on SGZ-enhanced low-light images:
  - mAP50 dropped substantially due to **Recall collapse**
- Likely because enhancement altered/discarded features the detector relied on.

## Contributors
- Jinyao Zhou — Robotics Institute, Carnegie Mellon University
- Yichen Ji — Heinz College, Carnegie Mellon University
- Xiaolei Hu — MCS, Carnegie Mellon University
- En Zheng — Department of Chemistry, Carnegie Mellon University

## Citation
\[1\] pepperpeople. All pepper datasets dataset. https://universe.roboflow.com/pepperpeople/all-pepper-datasets, oct 2023. visited on 2025-10-24.

\[2\] Shen Zheng and Gaurav Gupta. Semantic-guided zero-shot learning for low-light image/video enhancement. In Proceedings of the IEEE/CVF Winter conference on applications of computer
vision, pages 581–590, 2022.

\[3\] Mingbo Hong, Shen Cheng, Haibin Huang, Haoqiang Fan, and Shuaicheng Liu. You only look around: Learning illumination-invariant feature for low-light object detection. In A. Globerson,
L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 87136–87158. Curran Associates,
Inc., 2024.

\[4\] Ayan Paul and Rajendra Machavaram. Advancing capsicum detection in night-time greenhouse environments using deep learning models: Comparative analysis and improved zero-shot
detection through fusion with a single-shot detector. Franklin Open, 10:100243, 2025. ISSN 2773-1863.
