import numpy as np
import os.path as path

import pandas as pd

from pykeen.triples import TriplesFactory
from pykeen.hpo import hpo_pipeline
from pykeen.models.unimodal import TransE
from pykeen.training import SLCWATrainingLoop

from kgrec.datasets import Dataset
from kgrec.kg.entities import get_entities
from kgrec.kg.statements import collect_statements


def get_model_name(k: int, scoring_fct_norm: int,
                   epochs: int, batch_size: int, loss_margin: float,
                   num_negs_per_pos: int, lr_optimizer: float, seed: int):
    lm_str = str(loss_margin).replace('.', '_')
    lro_str = str(lr_optimizer).replace('.', '_')
    return 'transE_k%d_sc%d_e%d_bs%d_lm%s_np%d_lro%s_s%d.tsv' % \
           (k, scoring_fct_norm, epochs, batch_size, lm_str, num_negs_per_pos,
            lro_str, seed)


def train(dataset: Dataset, model_out_directory: str, k: int,
          scoring_fct_norm: int, epochs: int, batch_size: int,
          loss_margin: float, num_negs_per_pos: int, lr: float, seed: int):
    kg = np.array(collect_statements(dataset), dtype=str)
    kg = TriplesFactory.from_labeled_triples(triples=kg,
                                             create_inverse_triples=False)

    model = TransE(triples_factory=kg, embedding_dim=k,
                   scoring_fct_norm=scoring_fct_norm,
                   loss_kwargs=dict(margin=loss_margin),
                   random_seed=seed)

    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=kg,
        negative_sampler_kwargs=dict(num_negs_per_pos=num_negs_per_pos),
        optimizer_kwargs=dict(lr=lr)
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
                           get_model_name(k, scoring_fct_norm, epochs,
                                          batch_size, loss_margin,
                                          num_negs_per_pos, lr, seed))
    pd.DataFrame(embeddings).to_csv(model_file, index=False,
                                    sep='\t', header=False)


def train_hpo(dataset: Dataset, model_out_directory: str, trials: int,
              seed: int):
    kg = np.array(collect_statements(dataset), dtype=str)
    kg = TriplesFactory.from_labeled_triples(triples=kg,
                                             create_inverse_triples=False)

    training, testing, validation = kg.split([.8, .1, .1], random_state=seed)
    result = hpo_pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model='TransE',
        model_kwargs=dict(random_seed=seed),
        n_trials=trials,
    )
    result.save_to_directory(path.join(model_out_directory, dataset.name,
                                       'transE-hpo'))

    best_t = result.study.best_trial
    train(dataset, model_out_directory,
          k=best_t.params['model.embedding_dim'],
          scoring_fct_norm=best_t.params['model.scoring_fct_norm'],
          epochs=best_t.params['training.num_epochs'],
          batch_size=best_t.params['training.batch_size'],
          loss_margin=best_t.params['loss.margin'],
          lr=best_t.params['optimizer.lr'],
          num_negs_per_pos=best_t.params['negative_sampler.num_negs_per_pos'],
          seed=seed)
