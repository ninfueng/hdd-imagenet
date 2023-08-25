.PHONY: fmt
fmt:
	isort .
	black .

.PHONY: clean
clean:
	find -iname __pycache__ | xargs rm -rf
	rm -rf ./database

.PHONY: download
download:
	conda install -c conda-forge python-prctl
	conda install protobuf
	pip install --upgrade git+https://github.com/tensorpack/dataflow.git
	pip install python-prctl
	pip install lmdb
	pip install git+https://github.com/BayesWatch/sequential-imagenet-dataloader.git
