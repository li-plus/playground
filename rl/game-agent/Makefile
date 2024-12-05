.PHONY: lint

lint:
	autoflake -i --remove-all-unused-imports *.py
	isort *.py
	black *.py --line-length 120 --verbose
