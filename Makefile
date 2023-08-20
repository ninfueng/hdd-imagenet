.PHONY: fmt
fmt:
	isort .
	black .

.PHONY: clean
clean:
	find -iname __pycache__ | xargs rm -rf
	rm -rf ./database
