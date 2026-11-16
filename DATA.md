# Data and licences

Nothing from the datasets is stored in this repository. Images, checkpoints and
logs are git-ignored. Everything is downloaded by script.

## State Farm Distracted Driver Detection (Kaggle)

- Source: https://www.kaggle.com/c/state-farm-distracted-driver-detection
- Content: labelled 2D dashboard-camera images of drivers, 10 classes (c0 to c9),
  plus `driver_imgs_list.csv` mapping each image to a driver (`subject`).
- How to get it: accept the competition rules on Kaggle, then `make data`.
- Licence / terms of use (Rules tab, read on 2026-11-16; my reading, not legal advice):
  - No licence is granted on the data. Downloading it means accepting the
    competition rules as a binding agreement with State Farm.
  - Use outside the competition: the rules allow use "solely for the purpose and
    duration of the Competition" (it ended on 2016-08-01), unless the competition
    website says otherwise. Use for a personal learning project after the end is
    therefore not explicitly covered. Nothing distinguishes commercial from
    non-commercial use, because no use beyond the competition is granted at all.
  - Redistribution: forbidden. The data must not be transmitted, duplicated,
    published or made available to anyone who has not accepted the rules, and
    access by such people must be prevented.
  - Sample images: they are part of the data, so showing them publicly counts as
    publishing it.
  - Code: code based on the data may be shared publicly (not privately), and
    sharing it publicly places it under the MIT License.
  - Trained weights: not mentioned by the rules.
  - Data tab (read on 2026-11-17): it describes the classes and the files and adds
    no terms of use, so the Rules tab is the only source of conditions.
- Facts from the Data tab that matter for the evaluation:
  - The images were staged in a controlled environment: a truck towed the car, so
    the "drivers" were not actually driving.
  - Kaggle's own train/test split is by driver, like the splits in this repo.
  - Metadata such as creation dates was removed.
  - The unlabelled test set contains extra resized images that are ignored in
    scoring, so it is not a clean sample of the same distribution.
- Consequences for this repo:
  - No images and no `driver_imgs_list.csv` are committed; `data/` is git-ignored
    and everything is downloaded by script with each user's own Kaggle account.
  - No sample images in the README, reports or notebooks; `reports/samples/` is
    git-ignored. Public reports contain only aggregate counts, metrics and plots.
  - No trained weights are published, as a conservative choice.
  - The code is released under the MIT License.
  - The project is for personal study and portfolio only, not for commercial use.

## Optional extensions (only if time allows)

- Drive&Act (Fraunhofer IOSB / KIT): copyright Fraunhofer IOSB, "usage for research
  only". Code and metrics only in this repo, no frames and no weights.
- DMD (Vicomtech): CC BY-NC-ND 4.0, academic use only, RGB only at the time of
  writing. Not used.

## Privacy

The images show real people. Keep public examples to the minimum needed, and see
MODEL_CARD.md for intended use and limits.
