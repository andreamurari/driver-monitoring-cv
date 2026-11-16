# Data and licences

Nothing from the datasets is stored in this repository. Images, checkpoints and
logs are git-ignored. Everything is downloaded by script.

## State Farm Distracted Driver Detection (Kaggle)

- Source: https://www.kaggle.com/c/state-farm-distracted-driver-detection
- Content: labelled 2D dashboard-camera images of drivers, 10 classes (c0 to c9),
  plus `driver_imgs_list.csv` mapping each image to a driver (`subject`).
- How to get it: accept the competition rules on Kaggle, then `make data`.
- Licence / terms of use: **TODO (day 1)** copy here, in your own words, what the
  Rules and Data tabs allow: use outside the competition, commercial vs
  non-commercial, redistribution, showing sample images publicly.
  Date read: **TODO**
- Consequences for this repo: **TODO** (for example: no images committed, no
  trained weights published, sample images shown or not).

## Optional extensions (only if time allows)

- Drive&Act (Fraunhofer IOSB / KIT): copyright Fraunhofer IOSB, "usage for research
  only". Code and metrics only in this repo, no frames and no weights.
- DMD (Vicomtech): CC BY-NC-ND 4.0, academic use only, RGB only at the time of
  writing. Not used.

## Privacy

The images show real people. Keep public examples to the minimum needed, and see
MODEL_CARD.md for intended use and limits.
