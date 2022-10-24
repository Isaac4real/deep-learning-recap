make install:
	python -m pip install --upgrade pip poetry
	poetry config virtualenvs.create false
	poetry install --no-dev