PYTHONPATH := src/python
GO_CACHE := /tmp/ares-go-cache

.PHONY: test test-python test-go results report asm-benchmark cpp-build clean

test: test-python test-go

test-python:
	PYTHONPATH=$(PYTHONPATH) python3 -m unittest discover -s tests -v

test-go:
	GOCACHE=$(GO_CACHE) go test ./...

results:
	PYTHONPATH=$(PYTHONPATH) python3 -m ares.experiments

report:
	PYTHONPATH=$(PYTHONPATH) python3 -m ares.reporting

asm-benchmark:
	PYTHONPATH=$(PYTHONPATH) python3 src/assembly/benchmark_assembly.py

cpp-build:
	cmake -S src/planning -B build/planning
	cmake --build build/planning

clean:
	rm -rf build
