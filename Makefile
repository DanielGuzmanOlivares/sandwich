.PHONY: setup create-conda-env install-deps clean

CONDA_ENV=sandwich-env
PYTHON_VERSION=3.11

setup: create-conda-env install-deps unzip

# Create the Conda environment
create-conda-env:
	@echo "Creating Conda environment: $(CONDA_ENV)..."
	conda create -n $(CONDA_ENV) python=$(PYTHON_VERSION) ipython --yes

# Activate Conda and install Poetry dependencies
install-deps:
	@echo "Activating Conda environment and installing dependencies with Poetry..."
	@eval "$$(conda shell.bash hook)" && conda activate $(CONDA_ENV) && poetry install

unzip:
	@echo "Unzipping data..."
	unzip ./data/babelnet.zip -d ./data/
	unzip ./data/benchmarks.zip -d ./data
	rm data/__MACOSX -r

# Clean up poetry lock file
clean:
	@echo "Cleaning up..."
	rm -rf poetry.lock
