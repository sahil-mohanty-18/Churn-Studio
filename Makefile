.PHONY: setup download train run test smoke
setup:
	python run.py setup
download:
	python run.py download
train:
	python run.py train
run:
	python run.py serve
test:
	python run.py test
smoke:
	python tests/live_smoke.py
