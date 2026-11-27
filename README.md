# Driver monitoring: distraction detection from in-cabin images

Work in progress (day 8 of 14). Distraction classification on the State Farm Distracted
Driver dataset, evaluated on drivers never seen in training, plus a phone / cup / hand /
face detector trained on images I annotated myself. Every number below is measured and
traceable to a config, a commit and a seed in [reports/experiments.md](reports/experiments.md).

| Macro-F1, unseen drivers | mAP@0.5, detector | CPU latency, best classifier |
| --- | --- | --- |
| **0.908** (ResNet18, 320 px) | **0.78** (YOLO11n, 254 annotated images) | **50 ms** per frame |

Classifier metrics are on the validation drivers (4 people never seen in training); the
held-out test drivers are used once, at the end of the project. Detector mAP is on its own
test split (41 annotated images of the 4 test drivers), mean of two seeds.
Latency: batch 1, 4 CPU threads, laptop on mains power.

## What the project shows

- **Leakage.** The same ResNet18 scores 99.6% accuracy with a random image split and 86.6%
  with a split by driver: frames of one driver are near duplicates, so a random split
  measures recognition of people already seen. Every number here uses the driver split.
- **Ablations, one change at a time, with seed noise measured first.** Two seeds of the
  same config differ by about 0.1 log-loss, so smaller differences are not reported as
  results. Resolution was the largest effect (224 → 320 px: log-loss 0.42 → 0.28);
  removing augmentation clearly hurt; horizontal flip with label swap and ResNet50 stayed
  within noise.
- **ResNet vs ViT.** ViT-S/16 (ImageNet-21k pretraining) beats ResNet18 at 224 px on both
  seeds (log-loss 0.36 / 0.35 vs 0.42 / 0.52) but costs twice the CPU latency (59 vs 27 ms)
  and three times the training time. ResNet18 at 320 px is both more accurate and faster.
- **Error analysis.** "Talking to the passenger" has recall 0.65, mostly confused with safe
  driving: a missed distraction, the costly error for a safety system. Left and right hands
  are almost never confused, because the camera sits on the passenger side and the two poses
  look very different (which is also why mirrored training images did not help). Of the 30
  most confident errors, 18 are true model errors (a phone held at chest height read as a
  cup, a hand in the hair read as reaching behind) and 9 come from one questionable label
  sequence. Grad-CAM suggests the model looks at the background, not the head, for the
  passenger class: a possible shortcut, to be tested with background masking.
  Details in [reports/error_analysis_v1.md](reports/error_analysis_v1.md).
- **Annotation, and a contaminated test set caught.** 254 images labelled in Label Studio
  (phone, cup, hand, face) in two batches. The second batch was pre-labelled by the first
  detector, including its test images: scored partly against its own predictions, the old
  model reached mAP@0.5:0.95 0.73 there, against 0.38 on independent labels. The test
  images were re-annotated, and test labels now never start from a model under evaluation.
  Annotating the same images twice also measured my own consistency (mean IoU 0.73), so
  mAP@0.5 is the main detector metric. Guidelines in
  [annotations/GUIDELINES.md](annotations/GUIDELINES.md).
- **Detector: data beats model size.** YOLO11n (2.6 M parameters) fine-tuned from COCO on
  CPU. A targeted batch of phone and cup images raised mAP@0.5 from 0.73 to 0.78 (two seeds
  each) and halved missed phones; 960 px input and the 3.6x larger YOLO11s gave no gain for
  2.3-2.8x the inference time. Errors are missed objects, not misplaced boxes, and most
  missed faces belong to one test driver (face in profile against a bright window).

## Results so far

Validation set: 4 drivers never seen in training, 4,347 images.

| Model | Input | Params | Val log-loss | Val accuracy | CPU latency (median) |
| --- | --- | --- | --- | --- | --- |
| ResNet18, random split (leakage) | 224 | 11.2 M | 0.017 | 0.996 | 27 ms |
| ResNet18 (2 seeds) | 224 | 11.2 M | 0.42 / 0.52 | 0.866 / 0.844 | 27 ms |
| ResNet50 | 224 | 23.5 M | 0.53 | 0.820 | 60 ms |
| ViT-S/16 (2 seeds) | 224 | 21.7 M | 0.36 / 0.35 | 0.903 / 0.896 | 59 ms |
| **ResNet18** | **320** | 11.2 M | **0.28** | **0.912** | 50 ms |

## Next

A two-branch classifier that
adds detector crops of hands and phone; background masking to test the shortcut; ONNX
export and int8 quantization; final runs with three seeds and a single evaluation on the
test drivers.

## Evaluation protocol

All splits are by driver (`subject`), never by image (`splits.json`: 18 train, 4
validation, 4 test drivers). `tests/test_splits.py` checks that no driver appears in two
splits. The annotated detection images follow the same split.

## Reproduce

```bash
pip install -e ".[dev,train]"
make data                                        # Kaggle token and accepted rules, see DATA.md
python -m dms.train --config configs/resnet18_320.yaml   # trained on Kaggle (T4)
python -m dms.evaluate --run runs/<run>          # per-class, per-driver metrics
python -m dms.benchmark                          # CPU latency
python -m dms.detection.prepare_yolo             # needs local labels, see annotations/
python -m dms.detection.train_yolo --config configs/yolo11n.yaml
```

## Data and licences

See [DATA.md](DATA.md). No dataset content is stored in this repository: no images, no
sample pictures, no annotation labels derived from the images and no trained weights.
Intended use and limits are in [MODEL_CARD.md](MODEL_CARD.md).

Third-party software: Ultralytics YOLO is licensed under AGPL-3.0; MediaPipe models under
Apache-2.0; timm and torchvision weights under their own licences.

The code is released under the [MIT License](LICENSE). The licence covers the code only,
not the dataset.
