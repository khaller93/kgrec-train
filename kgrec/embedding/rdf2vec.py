import os.path as path
import pandas as pd

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker

from kgrec.datasets import Dataset
from kgrec.kg.entities import get_entities
from kgrec.kg.statements import collect_statements


def get_model_name(epochs: int, walks: int, path_length: int,
                   with_reverse: bool, skip_type: bool, seed: int):
    return 'rdf2vec_e%d_w%d_d%d_withR%s_skipType%s_s%d.tsv' % \
           (epochs, walks, path_length, str(with_reverse).lower(),
            str(skip_type).lower(), seed)


def train(dataset: Dataset, model_out_directory: str, epochs: int, walks: int,
          path_length: int, with_reverse: bool, skip_type: bool, num_jobs: int,
          seed: int):
    ent = get_entities(dataset, model_out_directory)
    skip_p = {'www.w3.org/1999/02/22-rdf-syntax-ns#type'} if skip_type else {}

    kg = KG(skip_verify=True)
    for stmt in collect_statements(dataset, model_out_directory):
        if str(stmt[1]) in skip_p:
            continue
        sub = Vertex(stmt[0])
        obj = Vertex(stmt[1])
        pred = Vertex(stmt[2], predicate=True, vprev=sub, vnext=obj)
        kg.add_walk(sub, pred, obj)

    transformer = RDF2VecTransformer(
        Word2Vec(epochs=epochs),
        walkers=[RandomWalker(max_walks=walks, max_depth=path_length,
                              random_state=seed, with_reverse=with_reverse,
                              n_jobs=num_jobs)],
        verbose=1
    )

    embeddings, _ = transformer.fit_transform(kg, ent['iri'].values.tolist())
    model_file = path.join(model_out_directory, dataset.name.lower(),
                           get_model_name(epochs, walks, path_length,
                                          with_reverse, skip_type, seed))
    pd.DataFrame(embeddings).to_csv(model_file, index=False,
                                    sep='\t', header=False)
