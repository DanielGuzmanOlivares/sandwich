# 🥪 SANDWiCH: Semantical Analysis of Neighbours for Disambiguating Words in Context ad Hoc

This repository contains the inference code and models associated with our NAACL 2025 paper. This project was developed with the support of Bulil Technologies S.L. as part of my PhD research.

The released version of the code differs slightly from the one used in the paper, as some parts of the pipeline had to be adapted from proprietary software to open-source Python implementations. However, the benchmark results (see below) still demonstrate a significant improvement over the previous state-of-the-art and remain within the same magnitude as those reported in the paper.

## License Information
The models and code in this repository are distributed under an MIT license. However, the data that the code uses is a modification of a small part of BabelNet, which is licensed under the non-commercial BabelNet license (https://babelnet.org/full-license). These extracts are derived from BabelNet 5.3 offline indices. A copy of the BabelNet license is included in [data/babelnet_license.txt](https://github.com/DanielGuzmanOlivares/sandwich/data/babelnet_license.txt).

## Setup & Installation
To set up this project, you need **Conda (or Miniconda)** and the **Poetry** package manager. You can install them as follows:

```shell
# Install Miniconda (if not installed)
https://www.anaconda.com/docs/getting-started/miniconda/install

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

### **Important:**
Ensure that Conda is properly initialized by running:
```shell
conda init
```

Once these dependencies are installed, follow these steps:

1. **Create and set up the environment:**
    ```shell
    make
    ```
    This will:
    - Create a Conda environment named `sandwich-env` using Python 3.11.
    - Install the necessary dependencies via Poetry.
    - Unzip required files containing the knowledge graph, definitions, and model.

2. **Activate the Conda environment:**
    ```shell
    conda activate sandwich-env
    ```

3. **Download the model weights:**
    Download the model weights from [this link](https://drive.google.com/file/d/1BUqg68_kC_nYCBFidw7jBVv6FN_iVDcN/view?usp=sharing) and extract them into the project folder.

## Running the Benchmarks
To evaluate the model, use the following command:

```shell
python benchmarks.py [--all] [--dataset] [--gpu] [--batch_size N] [--pos]
```

### **Options:**
- `--all` : Run all benchmarks.
- `--[dataset]` : Run the model on a specific dataset.
- `--gpu` : Enable GPU acceleration.
- `--batch_size N` : Set batch size (for faster inference on GPU).
- `--pos` : Break down metrics by part of speech.

### **Example Usage:**
```shell
# Run all benchmarks using GPU with batch size 16
python benchmarks.py --all --gpu --batch_size 16

# Run only the Semeval 2010 dataset
python benchmarks.py --dataset semeval2010
```

## Results

### **Performance on Semeval Datasets:**
| Semeval 2007 | Semeval 2002 | Semeval 2003 | Semeval 2010 | Semeval 2013 | Semeval 2015 | 42D  | ENSoft | ENHard |
|--------------|--------------|--------------|--------------|--------------|--------------|------|--------|--------|
| 80.9         | 87.8         | 85.7         | 88.2         | 92.6         | 91.2         | 80.3 | 90.0   | 50.6   |

### **Results on Raganato et al. 2017 Benchmark (by POS):**
| Total | Nouns | Verbs | Adjectives | Adverbs |
|-------|-------|-------|------------|---------|
| 88.6  | 93.2  | 75.7  | 85.9       | 88.7    |

These results confirm that the model significantly outperforms previous approaches across multiple datasets and parts of speech.

## Cite
If you use this code, please cite our paper:
<script src="https://gist.github.com/DanielGuzmanOlivares/06da02744aedb243dfa6c39fed90cc6e.js"></script>



