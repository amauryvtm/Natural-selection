.PHONY: install run test render clean all

PYTHON := python3

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m src.core
	$(PYTHON) -m src.render

test:
	$(PYTHON) -m pytest tests/tests.py -v

render:
	$(PYTHON) -m src.render

clean:
	rm -f data/work/history.npz data/work/stats.json data/work/evolution_4d.mp4 data/work/evolution_4d.gif data/work/stats.png

all: install run
