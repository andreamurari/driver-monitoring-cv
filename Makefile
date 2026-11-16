.PHONY: install data splits eda test

install:
	pip install -e ".[dev]"

data:
	bash scripts/download_statefarm.sh data/raw

splits:
	python scripts/make_splits.py --csv data/raw/driver_imgs_list.csv --out splits.json

eda:
	python scripts/eda.py --csv data/raw/driver_imgs_list.csv --imgs data/raw/imgs/train

test:
	pytest -q
