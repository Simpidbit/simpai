.PHONY: all build test

all: build test

build:
	python3 -m pip install -e .

test:
	python3 ./test/main.py

