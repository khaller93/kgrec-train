# KG-based recommendation

This application aims to enable experiments with KG-based recommendation. It
provides a command line interface to train models.

## Installation

A modern Python version (>=3.8) is needed for this application. The
required Python packages are specified in the `requirements.txt` and can be
installed with pip.

```bash
$ python -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```

Then, the application can be run in this environment by executing
the main Python file.

```bash
(venv) $ python main.py --help 
```

## Usage

This application can be used to train embeddings.

### Specify custom dataset

A custom dataset can be added by editing the `datasets.py` file in the
root directory. The application is designed to access the knowledge graph
over TSV files. The following files are expected:
* a file named `index.tsv.gz`, which is the index map of number to IRI 
compressed with GZip.
* a file named `statements.tsv.gz`, which is a list of statements (i.e. triples)
of the KG with the index number representing the actual IRI of entities. It only
includes statements with resources (i.e. no literals) as objects.

Optionally, these file can be provided:
* a file named `relevant_entities.tsv.gz`, which is a list of relevant entities.

### Train/Compute Models

| **Embeddings** | **Similarity Metrics**                                     |
| -------------- |------------------------------------------------------------|
| rdf2vec        | *Linked Data Semantic Distance (needs to be implemented!)* |
| transE         |                                                            |

#### Embeddings

**(A) Train rdf2vec Embedding**

```
Usage: main.py train rdf2vec [OPTIONS]

  train rdf2vec embeddings

Options:
  --epochs INTEGER                [default: 10]
  --walks INTEGER                 [default: 200]
  --path-length INTEGER           [default: 4]
  --with-reverse / --no-with-reverse
                                  [default: no-with-reverse]
  --skip-type / --no-skip-type    [default: no-skip-type]
  --seed INTEGER                  [default: 133]
  --model-out-directory TEXT      [default: model]
  --dataset TEXT                  [default: pokemon]

```

**(B) Train transE Embedding**

```
Usage: main.py train transe [OPTIONS]

  train transE embeddings

Options:
  --dim INTEGER               [default: 64]
  --scoring-fct-norm INTEGER  [default: 1]
  --epochs INTEGER            [default: 10]
  --batch-size INTEGER        [default: 256]
  --loss-margin FLOAT         [default: 0.5]
  --negs-per-pos INTEGER      [default: 4]
  --seed INTEGER              [default: 133]
  --model-out-directory TEXT  [default: model]
  --dataset TEXT              [default: pokemon]
```

**(C) Hyperparameter Tuning for transE**

Parameters can be tuned for transE using link-prediction train, test and
validation sets. Then we can assume that a good parameter for link prediction
can also be used for similarity/relatedness.

```
Usage: main.py train transe-hpo [OPTIONS]

  tune parameters and train transE embeddings with best found parameters

Options:
  --trials INTEGER            [default: 10]
  --seed INTEGER              [default: 133]
  --model-out-directory TEXT  [default: model]
  --dataset TEXT              [default: pokemon]
```

#### Similarity Metrics

The following metrics require the graph database Neo4J to be computed. The
application will try to use Docker to start a Neo4J instance and import the data
for processing. Hence, Docker needs to be running, and the current user
executing this application needs access to the Docker interface.

**(1) Compute LDSD Metric**

The similarity metric LDSD hasn't any hyperparameter to choose.

```
Usage: main.py compute ldsd [OPTIONS]

  compute LDSD similarity metric

Options:
  --model-out-directory TEXT  [default: model]
  --dataset TEXT              [default: pokemon]
```

The output of this computation is a pair of indices referring back to the 
`entities.tsv.gz` index, and the computed value of this pair. If a pair isn't
contained in the output, then its assumed that the value is `1.0`, which
indicates no similarity.

## Contact

* Kevin Haller - [contact@kevinhaller.dev](mailto:contact@kevinhaller.dev)
