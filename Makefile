.PHONY: format

format:
	isort format --skip wandb
	autoflake -r --exclude wandb --in-place --remove-unused-variables --remove-all-unused-imports format
	black -l 120 --exclude wandb format