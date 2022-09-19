# KG-based recommendation

This application should allow to play around with KG-based recommendation. It
aims to provide a command line interface to train models, analyse the results
and recommend items based on a user-item KG.

## Installation

A modern Python version (>=3.9) is needed for this application. The
required Python packages are specified in the `requirements.txt` and can be
installed with pip.

```bash
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Then, the application can be run in this environment by executing
the main Python file.

```bash
(venv) $ python main.py --help 
```

## Usage

This application can be used to train embeddings, compute similarity metrics
and analyse the 

### Specify custom dataset

A custom dataset can be added by editing the `datasets.py` file in the
root directory. The application is designed to access the knowledge graph
over a SPARQL endpoint.

Thus, a dataset is defined by a SPARQL endpoint URL. Optionally, also the
default named graph can be specified as well as named graphs to ignore in the
fetching of entities and statements.

### Train RDF2Vec Embedding

```
Usage: main.py rdf2vec [OPTIONS]

  train the rdf2vec embeddings

Options:
  --epochs INTEGER            [default: 10]
  --walks INTEGER             [default: 200]
  --path-length INTEGER       [default: 4]
  --seed INTEGER              [default: 133]
  --model-out-directory TEXT  [default: model]
  --dataset TEXT              [default: pokemon]
```

### Train TransE Embedding

```
Usage: main.py transe [OPTIONS]

Options:
  --k INTEGER                 [default: 64]
  --epochs INTEGER            [default: 10]
  --batch-size INTEGER        [default: 256]
  --seed INTEGER              [default: 133]
  --model-out-directory TEXT  [default: model]
  --dataset TEXT              [default: pokemon]
```

### Compute LDSD Metric

The similarity metric LDSD hasn't any hyperparameter to choose.

```
Usage: main.py ldsd [OPTIONS]

Options:
  --model-out-directory TEXT  [default: model]
  --dataset TEXT              [default: pokemon]
```

## Contact

* [Kevin Haller](mailto://contact@kevinhaller.dev)