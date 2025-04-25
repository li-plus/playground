.PHONY: lint

lint:
	autoflake -i --remove-all-unused-imports hf/*.py pytorch/*.py py/*.py ray/*.py
	isort hf/ pytorch/ py/ ray/
	black hf/ pytorch/ py/ ray/ --verbose --line-length 120
	clang-format -i **/*.cu
