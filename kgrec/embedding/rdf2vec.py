import logging

import pandas as pd
import sqlitekg2vec

from kgrec.datasets import Dataset
from kgrec.embedding.embedding import Embedding
from multiprocessing import cpu_count
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
            logging.info('constructing RDF2Vec specific KG for "%s" with: '
                         '{skip_type: %s}', self.dataset.name, skip_type)
            with sqlitekg2vec.open_from(
                    self.dataset.statement_iterator(),
                    skip_predicates=skip_p,
            ) as kg:
                # construct the transformer
                logging.info('training RDF2Vec model for KG "%s" with: '
                             '{max_walks: %d, max_depth: %d, with_reverse: %s, '
                             'seed: %d, n_jobs: %d}', self.dataset.name, walks,
                             path_length, with_reverse, seed, num_jobs)
                transformer = RDF2VecTransformer(
                    Word2Vec(epochs=epochs),
                    walkers=[RandomWalker(max_walks=walks,
                                          max_depth=path_length,
                                          random_state=seed,
                                          with_reverse=with_reverse,
                                          n_jobs=num_jobs)],
                    verbose=1
                )
                # train RDF2Vec
                ent_names = [str(key) for key, _ in self.dataset.index(
                    check_for_relevance=True).forward.items()]
                ent = kg.entities(restricted_to=ent_names)
                embeddings, _ = transformer.fit_transform(kg, ent)
                logging.info('writing trained RDF2Vec model for KG "%s"',
                             self.dataset.name)
                packed_embed = kg.pack(ent, embeddings)
                return pd.DataFrame([vec for _, vec in packed_embed],
                                    index=[key for key, _ in packed_embed])

        return train(**model_kwargs)
