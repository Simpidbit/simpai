.PHONY: all build test

all: build test

build:
	python3 -m pip install -e .

test:
	cd ./test/vdsr && python3 train.py && cd ../../

