# ConANN: Conformal Approximate Nearest Neighbor Search

## Content
- `conann/`: extensions of faiss-1.9.0 with ConANN modifications to IndexIVF.
- `conann/eval`: parameterised experiment executors for ConANN
- `faiss-1.9.0/`: fork form [github/facebookresearch/faiss](https://github.com/facebookresearch/faiss) version 1.9.0
- `faiss-1.9.0/eval`: parameterised experiment executors for faiss-1.9.0
- `auncel/`: faithful port of [github/pkusys/Auncel](https://github.com/pkusys/Auncel) from faiss-1.15.2 to faiss-1.9.0
- `auncel/eval`: parameterised experiment executors for Auncel

## Prerequisites

The experiments were conducted on Google Cloud Platform (GCP) using a virtual machine with an Intel(R) Xeon(R) CPU @ 2.80GHz, 64 vCPUs, 256 GB RAM, and Ubuntu 20.04.6 LTS. To achieve comparable performance the Intel-MKL SIMD instruction set needs to be installed on the machine.

## Datasets

The data sources can be found here:
- SIFT (1M): http://corpus-texmex.irisa.fr 
- GIST (1M): http://corpus-texmex.irisa.fr
- Fasttext (1M): https://huggingface.co/fse/fasttext-wiki-news-subwords-300
- DEEP (10M): https://disk.pku.edu.cn/link/AAD0A67DE2E7984DB5B5D4885871219AEF
- Bert (30522): The Bert embeddings have been extracted from the pre-trained model available on HuggingFace using the following code snippet:
```
from transformers import BertModel

def get_bert_emb():
    model_name = "bert-base-uncased"
    model = BertModel.from_pretrained(model_name)
    word_embeddings = model.embeddings.word_embeddings.weight
    word_embeddings_np = word_embeddings.detach().cpu().numpy()
    print("Word Embedding Shape:", word_embeddings.shape)
    return word_embeddings_np
```
---

The datasets should be placed in a folder named `data/<dataset_name>`. File format should be `.fvecs`, a simple binary format for storing float vectors.
Apart from the training set the following files are required:
- `queries.fvecs`
- `distances-10.fvecs`
- `distances-100.fvecs`
- `distances-1000.fvecs`
- `indices-10.fvecs`
- `indices-100.fvecs`
- `indices-1000.fvecs`
> Check `conann/conann/eval/error.cpp` for exact naming if data loading problems occur.

To create the query sample and ground truths using `experiments/experiment_controller.py` is recommended, as it contrains convenience methods for this purpose. The following programs need to be compiled first:
```
cd Auncel
cmake -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -B build .
make -C build -j sample_queries
make -C build -j compute_gt
```

## Build

Compile all the executors:

Auncel:

```
cd Auncel
cmake -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -B build .
make -C build -j sample_queries
make -C build -j compute_gt
make -C build -j effect_error
```

Faiss:

```
cd faiss-1.9.0
cmake -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -B build .
make -C build -j error
```

ConANN:

```
cd conann
cmake -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -B build .
make -C build -j error
make -C build -j latency
```

### Experiments with variable k
Because enabling experiments where k is no longer a fixed value required significant changes to core components in the code, they reside on a different git branch called: `variable-k`.

ConANN (variable k):

```
git checkout variable-k
cd conann
cmake -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -B build .
make -C build -j variable_k
```

Faiss (variable k):

```
git checkout variable-k
cd faiss-1.9.0
cmake -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -B build .
make -C build -j variable_k
```

## Experiments

All experiment setups are documented in and can be executed using the `experiment-controller.py` script.

```
cd experiments
python3 experiment-controller.py
```