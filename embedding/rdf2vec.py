import os.path as path
import pandas as pd

from multiprocessing import cpu_count

from datasets import Dataset
from kg import entities
from pyrdf2vec.graphs import KG
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker


def get_model_name(epochs: int, walks: int, path_length: int, seed: int):
    return 'rdf2vec_e%d_w%d_d%d_s%d.tsv' % (epochs, walks, path_length, seed)


def train(epochs: int, walks: int, path_length: int, seed: int,
          model_out_directory: str, dataset: Dataset):
    ent = entities.get_entities(dataset, model_out_directory)
    kg = KG(
        dataset.sparql_endpoint,
        skip_predicates={'www.w3.org/1999/02/22-rdf-syntax-ns#type'},
        literals=[],
        skip_verify=True,
    )

    transformer = RDF2VecTransformer(
        Word2Vec(epochs=epochs),
        walkers=[RandomWalker(max_walks=walks, max_depth=path_length,
                              random_state=seed, with_reverse=False,
                              n_jobs=cpu_count())],
        verbose=1
    )

    embeddings, _ = transformer.fit_transform(kg, ent['iri'].values.tolist())
    model_file = path.join(model_out_directory, dataset.name.lower(),
                           get_model_name(epochs, walks, path_length, seed))
    pd.DataFrame(embeddings).to_csv(model_file, index=False,
                                    sep='\t', header=False)
