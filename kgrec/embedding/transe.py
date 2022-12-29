import logging

import pandas as pd
import torch

from kgrec.datasets import Dataset
from kgrec.embedding.embedding import Embedding
from os import makedirs
from os.path import join, exists
from pykeen.hpo import hpo_pipeline
from pykeen.models.unimodal import TransE
from pykeen.triples import TriplesFactory
from pykeen.training import SLCWATrainingLoop
from pykeen.utils import resolve_device


class TransEModel(Embedding):
    """ a TransE embedding that can be trained on KG """

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    def _get_name(self) -> str:
        return 'transE'

    def _perform_training(self, model_kwargs, training_kwargs) -> pd.DataFrame:
        def train(dim: int, scoring_fct_norm: int, epochs: int, batch_size: int,
                  loss_margin: float, num_negs_per_pos: int,
                  optimizer_lr: float, seed: int):
            # construct the KG for training
            stmt_df = self.dataset.statements
            logging.info(
                'constructing triple factory from KG "%s" (size: %d) for '
                'transE learning', self.dataset.name, len(stmt_df))
            kg = TriplesFactory.from_labeled_triples(
                triples=stmt_df.applymap(
                    lambda x: str(x)).values,
                create_inverse_triples=False)
            # construct the learning model loop
            logging.info(
                'constructing transE model for "%s" with: {dim: %d,'
                'scoring_fct_norm: %d, loss_margin: %f, seed: %d}',
                self.dataset.name, dim, scoring_fct_norm, loss_margin, seed)
            model = TransE(triples_factory=kg, embedding_dim=dim,
                           scoring_fct_norm=scoring_fct_norm,
                           loss_kwargs=dict(margin=loss_margin),
                           random_seed=seed)

            if torch.has_cuda:
                _device: torch.device = resolve_device('gpu')
                model.to(_device)
                logging.info(
                    'selected device "CUDA" for transE learning for "%s"',
                    self.dataset.name)
            elif torch.has_mps:
                logging.info(
                    'selected device "Apple Metal GPU" for transE learning'
                    ' for "%s"',
                    self.dataset.name)
                _device: torch.device = resolve_device('mps')
                model.to(_device)
            else:
                logging.info(
                    'selected device "CPU" for transE learning for "%s"',
                    self.dataset.name)

            logging.info(
                'constructing transE training loop for "%s" with: '
                '{num_negs_per_pos: %d, optimizer_lr: %f}',
                self.dataset.name, num_negs_per_pos, optimizer_lr)
            training_loop = SLCWATrainingLoop(
                model=model,
                triples_factory=kg,
                negative_sampler_kwargs=dict(num_negs_per_pos=num_negs_per_pos),
                optimizer_kwargs=dict(lr=optimizer_lr)
            )

            # train the model
            logging.info(
                'trains transE model for "%s" with: {num_epochs: %d, '
                'batch_size: %d}', self.dataset.name, epochs, batch_size)
            _ = training_loop.train(
                triples_factory=kg,
                num_epochs=epochs,
                pin_memory=True,
                batch_size=batch_size,
            )
            # fetch the embeddings
            logging.info(
                'fetches and writes transE embeddings for "%s"',
                self.dataset.name)
            if torch.has_cuda or torch.has_mps:
                model.cpu()

            tensor = model.entity_representations[0](indices=None) \
                .detach().numpy()

            keys = []
            embeddings = []
            for entity_key, _ in self.dataset.index_iterator(
                    check_for_relevance=True):
                embeddings.append(tensor[kg.entity_to_id[str(entity_key)]])
                keys.append(entity_key)

            return pd.DataFrame(embeddings, index=keys)

        return train(**model_kwargs)


class TransEHPO:
    """ hyper-parameterization of TransE embedding on KG """

    def __init__(self, dataset: Dataset, model_out_directory: str):
        """
        creates a new optimizer for parameters of TransE model for the specified
        dataset.

        :param dataset: for which good parameters shall be searched.
        :param model_out_directory:
        """
        self.dataset = dataset
        self.model_out_directory = model_out_directory

    def _get_study_dir(self) -> str:
        """
        gets the directory where to store the study results.

        :return: path to the the directory where to store the study results.
        """
        return join(self.model_out_directory, self.dataset.name,
                    'hpo', 'transE')

    def hpo(self, trials: int, seed: int,
            batch_size: int = None) -> TransEModel:
        """
        do hyper-parameterization with the specified number of trials, and then
        computes the TransE model for the best trial.

        :param trials: the number of attempts for finding a good transE model.
        :param seed: random seed for the split of the KG in train, test and
        validation set.
        :param batch_size: optionally setting the size of a batch to a specific
        value. It is `None` by default.
        :return: the TransE model of the best trial.
        """
        # construct the KG for training
        stmt = self.dataset.statements
        kg = TriplesFactory.from_labeled_triples(
            triples=stmt.applymap(lambda x: str(x)).values,
            create_inverse_triples=False)
        training, testing, validation = kg.split([.8, .1, .1],
                                                 random_state=seed)
        # run pipeline for hpo
        result = hpo_pipeline(
            training=training,
            testing=testing,
            validation=validation,
            model='TransE',
            training_kwargs={} if batch_size is None else {
                'batch_size': batch_size
            },
            model_kwargs=dict(random_seed=seed),
            n_trials=trials,
        )
        # write study results to disk
        study_out_dir = self._get_study_dir()
        if not exists(study_out_dir):
            makedirs(study_out_dir)
        result.save_to_directory(study_out_dir)
        # train transE on best trial parameters
        best_t = result.study.best_trial
        trans_e = TransEModel(self.dataset)
        bs = best_t.params['training.batch_size'] if batch_size is None else \
            batch_size
        trans_e.train(model_kwargs=dict(
            dim=best_t.params['model.embedding_dim'],
            scoring_fct_norm=best_t.params['model.scoring_fct_norm'],
            epochs=best_t.params['training.num_epochs'],
            batch_size=bs,
            loss_margin=best_t.params['loss.margin'],
            optimizer_lr=best_t.params['optimizer.lr'],
            num_negs_per_pos=best_t.params['negative_sampler.num_negs_per_pos'],
            seed=seed))
        return trans_e
