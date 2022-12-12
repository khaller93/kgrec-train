import pandas as pd

from kgrec.datasets import Dataset
from kgrec.embedding.embedding import Embedding
from multiprocessing import cpu_count
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker

type_pred = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'


class RDF2VecModel(Embedding):
    """ a RDF2Vec embedding that can be trained on KG """

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def _get_name(self) -> str:
        return 'rdf2vec'

    def _perform_training(self, model_kwargs, training_kwargs) -> pd.DataFrame:
        num_jobs = training_kwargs['num_jobs'] \
            if training_kwargs and 'num_jobs' not in training_kwargs else \
            cpu_count()

        def train(epochs: int, walks: int, path_length: int, with_reverse: bool,
                  skip_type: bool, seed: int):
            type_index = self.dataset.index().backward[type_pred]
            skip_p = {str(type_index)} if skip_type and type_index != -1 else {}
            # construct a KG for training
            kg = KG(skip_verify=True)
            for stmt in self.dataset.statement_iterator():
                if str(stmt[1]) in skip_p:
                    continue
                sub = Vertex(str(stmt[0]))
                obj = Vertex(str(stmt[1]))
                pred = Vertex(str(stmt[2]), predicate=True, vprev=sub,
                              vnext=obj)
                kg.add_walk(sub, pred, obj)
            # construct the transformer
            transformer = RDF2VecTransformer(
                Word2Vec(epochs=epochs),
                walkers=[RandomWalker(max_walks=walks, max_depth=path_length,
                                      random_state=seed,
                                      with_reverse=with_reverse,
                                      n_jobs=num_jobs)],
                verbose=1
            )
            # train RDF2Vec
            ent = [str(key) for key, _ in
                   self.dataset.index(check_for_relevance=True).forward.items()]
            embeddings, _ = transformer.fit_transform(kg, ent)
            return pd.DataFrame(embeddings)

        return train(**model_kwargs)
