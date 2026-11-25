# Experiment log

Each row points to a config, a commit and a seed. Classifier metrics are on the
**validation** drivers (4 people), which were also used to pick the best epoch, so they
are slightly optimistic. The test set is used once, at the end of the project.

| Date | Run | Config | Commit | Seed | Split | Best epoch | Val log-loss | Val acc | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-18 | resnet18_driver_s0 | configs/resnet18.yaml | 1b2228e | 0 | by driver | 7 | 0.4195 | 0.866 | baseline |
| 2026-11-18 | resnet18_random_s0 | configs/resnet18.yaml + `split: random` | 1b2228e | 0 | random by image | n/a | 0.0165 | 0.996 | leakage experiment only |
| 2026-11-18 | resnet18_driver_s1 | configs/resnet18_s1.yaml | a243f03 | 1 | by driver | 8 | 0.5156 | 0.844 | seed noise |
| 2026-11-18 | resnet18_aug-none_s0 | configs/resnet18_aug-none.yaml | a243f03 | 0 | by driver | 7 | 0.6820 | 0.778 | ablation: no augmentation |
| 2026-11-18 | resnet18_aug-flip_s0 | configs/resnet18_aug-flip.yaml | a243f03 | 0 | by driver | 8 | 0.4904 | 0.834 | ablation: flip, inconclusive |
| 2026-11-18 | resnet18_320_s0 | configs/resnet18_320.yaml | a243f03 | 0 | by driver | 7 | 0.2763 | 0.912 | best so far, new baseline |
| 2026-11-18 | resnet50_s0 | configs/resnet50.yaml | a243f03 | 0 | by driver | 4 | 0.5255 | 0.820 | overfits after epoch 4 |
| 2026-11-19 | vit_s16_s0 | configs/vit_s16.yaml | 8ef51d5 | 0 | by driver | 8 | 0.3600 | 0.903 | ViT recipe (lr 5e-5, wd 0.05) |
| 2026-11-19 | vit_s16_s1 | configs/vit_s16_s1.yaml | d61befa | 1 | by driver | 6 | 0.3468 | 0.896 | ViT seed noise |

Environment: Kaggle, 1× Tesla T4, batch 64, 224 letterbox, 3 head epochs (lr 1e-3) +
5 fine-tuning epochs (lr 1e-4), AdamW, cosine schedule, no label smoothing.
Full metrics in `reports/metrics/`.

## Day 2: baseline and leakage

- **Leakage:** same model and config, only the split changes. With the split by driver
  (18 drivers in training, 4 unseen in validation) accuracy is 0.866; with a random split
  by image it reaches 0.996. The second number is inflated: every validation driver also
  appears in training, with near-identical frames, so the model recognises instead of
  generalising. A real system sees new people: the number to report is 0.866.
- **Head phase vs fine-tuning:** with the backbone frozen, validation accuracy stops at
  about 0.51; after fine-tuning it reaches 0.866. ImageNet features alone cannot separate
  very similar postures (phone in the right or left hand, phone or drink): the backbone
  has to adapt and learn to look at hands and objects.
- **Overfitting:** train accuracy 1.000 vs validation 0.866. Validation loss is lowest at
  epoch 7 (0.4195) and rises at epoch 8 (0.4404); the saved checkpoint is epoch 7. With
  train accuracy already at 100%, more epochs would not help: more regularisation and
  augmentation are needed (day 3), not longer training.
- **Noise:** the 1+1-epoch smoke test had scored 0.885, more than the full run (0.866).
  With only 4 validation drivers a 2-point difference can depend on who those drivers
  are: a single number is not enough to compare two configurations. More seeds are
  needed and, later, a grouped cross-validation.
- **Timing:** about 60 s per epoch with the backbone frozen and 68 s with full
  fine-tuning, even though the second updates thousands of times more parameters. The
  bottleneck is not the GPU but data loading on the CPU (reading and decoding 640x480
  JPEGs, resizing, augmentation). Possible fix: pre-resize the images.

## Day 3: ablations

Each run changes one thing relative to the baseline `resnet18_driver_s0`.
Main metric: validation log-loss (accuracy for reference).

| Run | What changes | Prediction | Result |
|---|---|---|---|
| resnet18_driver_s1 | seed 1 | similar | similar: measures the noise |
| resnet18_aug-none_s0 | no augmentation | worse | worse |
| resnet18_aug-flip_s0 | standard + flip with label swap | better | inconclusive |
| resnet18_320_s0 | resolution 320 | better | better |
| resnet50_s0 | ResNet50 | better | refuted: not better |

- **Noise (seed 0 vs seed 1):** same config; only the head's initial weights, the batch
  order and the augmentation draws change. Log-loss goes from 0.420 to 0.516 and accuracy
  from 0.866 to 0.844. A difference of this size (about 0.1 log-loss, 2 accuracy points)
  is not enough to call one configuration better. Two seeds underestimate the true noise:
  important decisions need more.
- **Augmentation:** without augmentation, log-loss rises to 0.682 and accuracy drops to
  0.778, well beyond the noise: with only 18 training drivers the model memorises (train
  accuracy 1.000), and colour and geometric augmentation is necessary. Flip with label
  swap gives 0.490 log-loss (better than seed 1) but 0.834 accuracy (one point below both
  seeds): the two metrics disagree and the difference is small, so the result is
  inconclusive.
- **Resolution:** at 320 (letterbox, long side 320) log-loss drops to 0.276 and accuracy
  rises to 0.912, well beyond the noise. It is the clearest result of the day: small
  details such as the phone, hands and cup take few pixels and get lost at 224. Cost:
  about twice the pixels per image, so slower training and inference.
- **ResNet50:** the best checkpoint is at epoch 4, the first fine-tuning epoch (0.526);
  afterwards validation loss climbs to 0.612 while train accuracy approaches 1.0. It
  overfits: 23 million parameters vs 11, with the same 18 drivers and the same learning
  rate. Compared with seed 1 the difference is within the noise: the "better" prediction
  is refuted, but it is not shown to be worse. With this recipe a larger model does not
  help; it would need a lower learning rate, fewer epochs or more regularisation.

## Day 4: ViT-S/16 vs ResNet

ViT-S/16 (`vit_small_patch16_224.augreg_in21k_ft_in1k`, timm) with the same pipeline as
the 224 baseline. Each model gets its standard recipe: the normalisation of its own
pretraining (ViT: mean/std 0.5) and, for the ViT, fine-tuning with lr 5e-5 and weight
decay 0.05.

Prediction: uncertain. First "ResNet18 stays better, too little data for a ViT", then
"20,000 images might be enough for fine-tuning". For seed 1: similar to seed 0.

| Model | Resolution | Params | Val log-loss (seed 0 / 1) | Val acc (seed 0 / 1) | CPU latency, median (p95) | s/epoch fine-tuning (T4) |
|---|---|---|---|---|---|---|
| ResNet18 | 224 | 11.2 M | 0.420 / 0.516 | 0.866 / 0.844 | 27 ms (29) | ~68 |
| ViT-S/16 | 224 | 21.7 M | 0.360 / 0.347 | 0.903 / 0.896 | 59 ms (62) | ~188 |
| ResNet50 | 224 | 23.5 M | 0.526 / n/a | 0.820 / n/a | 60 ms (63) | n/a |
| ResNet18 | 320 | 11.2 M | 0.276 / n/a | 0.912 / n/a | 50 ms (53) | n/a |

Latency: `python -m dms.benchmark`, batch 1, 4 threads, 100 runs after 20 warm-up runs,
laptop on mains power (on battery the times double and fluctuate). Details in
`reports/latency.json`.

- **Accuracy:** at 224 the ViT beats ResNet18 on both seeds and both metrics: log-loss
  0.360 and 0.347 vs 0.420 and 0.516, accuracy 0.903 and 0.896 vs 0.866 and 0.844. With a
  single seed the difference from the ResNet (0.06) was within the noise; the second seed
  confirms it. The gap between the two seeds is also much smaller (0.013 vs 0.096). The
  "too little data for a ViT" prediction is refuted: ImageNet-21k pretraining makes up for
  the 18 training drivers. The best model is still ResNet18 at 320 (0.276, one seed), and
  the ViT at 320 has not been tried.
- **Head phase:** with the backbone frozen the ViT reaches 0.63-0.64 validation accuracy,
  ResNet18 about 0.51. The pretrained ViT features already suit the task better before
  fine-tuning.
- **Noisy curves:** the ViT's validation loss swings during fine-tuning (seed 1: 0.37,
  0.61, 0.35, 0.42, 0.38). The best checkpoint is picked on the same validation set, so
  the reported number is somewhat optimistic, more so than for smooth curves.
- **Latency and cost:** on CPU the ViT takes 59 ms per image, like ResNet50 (60 ms) with a
  similar parameter count, and twice ResNet18 (27 ms). The parameter count alone does not
  decide speed. In training the ViT costs almost 3 times ResNet18 on a T4 (188 vs 68 s per
  fine-tuning epoch): with the ResNet the bottleneck was data loading, with the ViT it is
  the GPU. ResNet18 at 320 is both more accurate (one seed) and faster (50 ms) than the
  ViT at 224.
- **Limits of the comparison:** different recipes (standard normalisation, lr and weight
  decay for each model), so the measured gap also includes hyperparameters; only 2 seeds;
  4 validation drivers; checkpoint chosen on the same validation set. The final word will
  come from the test set and a grouped cross-validation.

## Day 7: first detector

| Date | Run | Config | Commit | Seed | Data | Test mAP@0.5 | Test mAP@0.5:0.95 | AP@0.5 phone / cup / hand / face |
|---|---|---|---|---|---|---|---|---|
| 2026-11-25 | yolo11n_batch1_s0 | configs/yolo11n.yaml | 254326b | 0 | 146 annotated images (100 / 23 / 23) | 0.741 | 0.380 | 0.663 / 0.587 / 0.950 / 0.764 |

YOLO11n fine-tuned from COCO weights, 640 px, 60 epochs, trained on CPU in 25 minutes;
the best checkpoint (on validation) is evaluated once on the detector's test split (23
images of the 4 test drivers).

- **Prediction confirmed:** hands and faces best, phones and cups worst. Phones and cups
  are small, have few examples (8 and 7 in the test split) and cups are often white on a
  bright background.
- **Small test split:** one phone found or missed moves its AP a lot; these numbers are a
  first estimate.
- **Faces:** AP 0.96 on validation but 0.76 on test; to be checked per driver.
- **Still improving:** validation mAP was still rising at epoch 60.

## Next steps

- [x] Day 3: ResNet50, augmentation ablation, 224 vs 320
- [x] Save the best epoch and per-epoch history in `metrics.json`
- [x] Print the device when `train.py` starts
- [ ] Use 320 as the new baseline for later comparisons
- [ ] ResNet50 with a lower fine-tuning lr (e.g. 3e-5) or fewer epochs
- [ ] More seeds for flip before deciding whether to keep it
- [x] Measure CPU latency at 224 and 320
- [ ] ViT at 320 (interpolated position embeddings, `img_size=320` in timm): latency and accuracy
- [ ] Second seed for ResNet18 at 320, before calling it the best model
- [ ] `--note` argument in `benchmark.py`, to record measurement conditions in the JSON
- [ ] Kaggle zip cell: rename the file every day, without `-j`
- [ ] Detector: per-driver metrics; targeted annotation batch for phones and cups
