import numpy as np
import os.path as path

import pandas as pd

from datasets import Dataset
from pykeen.triples import TriplesFactory
from pykeen.hpo import hpo_pipeline
from pykeen.models.unimodal import TransE
from pykeen.training import SLCWATrainingLoop

from kg.entities import get_entities
from kg.statements import collect_statements


def get_model_name(k: int, epochs: int, batch_size: int, seed: int):
    return 'transE_k%d_e%d_bs%d_s%d.tsv' % (k, epochs, batch_size, seed)


def train(dataset: Dataset, model_out_directory: str, k: int, epochs: int,
          batch_size: int, seed: int):
    kg = np.array(collect_statements(dataset.sparql_endpoint), dtype=str)
    kg = TriplesFactory.from_labeled_triples(triples=kg,
                                             create_inverse_triples=False)

    model = TransE(triples_factory=kg, embedding_dim=k)
    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=kg,
    )

    _ = training_loop.train(
        triples_factory=kg,
        num_epochs=epochs,
        batch_size=batch_size,
    )

    ent = get_entities(dataset, model_out_directory)

    entity_embedding_tensor = model.entity_representations[0](
        indices=None).detach().numpy()

    embeddings = []
    for entity in ent['iri']:
        embeddings.append(entity_embedding_tensor[kg.entity_to_id[entity]])

    model_file = path.join(model_out_directory, dataset.name.lower(),
                           get_model_name(k, epochs, batch_size, seed))
    pd.DataFrame(embeddings).to_csv(model_file, index=False,
                                    sep='\t', header=False)


def train_hpo(dataset: Dataset, model_out_directory: str, trials: int,
              seed: int):
    kg = np.array(collect_statements(dataset.sparql_endpoint), dtype=str)
    kg = TriplesFactory.from_labeled_triples(triples=kg,
                                             create_inverse_triples=False)

    training, testing, validation = kg.split([.8, .1, .1], random_state=seed)
    result = hpo_pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model='TransE',
        n_trials=trials,
    )
    result.save_to_directory(path.join(model_out_directory, dataset.name,
                                       'transE-hpo'))
