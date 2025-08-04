deps:
	uv sync

install:
	uv pip install .

test:
	pytest --ignore=tests/client/test_prerelease.py
