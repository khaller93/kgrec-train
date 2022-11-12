import pandas as pd
import torch

from kgrec.datasets import Dataset
from kgrec.embedding.embedding import Embedding
from kgrec.subsampling import SubSamplingFunc, id_sub
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
            kg = TriplesFactory.from_labeled_triples(
                triples=self.dataset.statements.applymap(
                    lambda x: str(x)).values,
                create_inverse_triples=False)
            # construct the learning model loop
            model = TransE(triples_factory=kg, embedding_dim=dim,
                           scoring_fct_norm=scoring_fct_norm,
                           loss_kwargs=dict(margin=loss_margin),
                           random_seed=seed)

            if torch.has_cuda:
                _device: torch.device = resolve_device('gpu')
                model.to(_device)
            elif torch.has_mps:
                _device: torch.device = resolve_device('mps')
                model.to(_device)

            training_loop = SLCWATrainingLoop(
                model=model,
                triples_factory=kg,
                negative_sampler_kwargs=dict(num_negs_per_pos=num_negs_per_pos),
                optimizer_kwargs=dict(lr=optimizer_lr)
            )
            # train the model
            _ = training_loop.train(
                triples_factory=kg,
                num_epochs=epochs,
                pin_memory=True,
                batch_size=batch_size,
            )
            # fetch the embeddings
            if torch.has_cuda or torch.has_mps:
                model.cpu()

            tensor = model.entity_representations[0](indices=None) \
                .detach().numpy()

            embeddings = []
            for entity in self.dataset.relevant_entities['key_index'].values:
                embeddings.append(tensor[kg.entity_to_id[str(entity)]])

            return pd.DataFrame(embeddings)

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
            subsampling: SubSamplingFunc = id_sub) -> TransEModel:
        """
        do hyper-parameterization with the specified number of trials, and then
        computes the TransE model for the best trial.

        :param trials: the number of attempts for finding a good transE model.
        :param seed: random seed for the split of the KG in train, test and
        validation set.
        :param subsampling: an optional function performing subsampling on the
        input dataset. Per default, the whole KG is used.
        :return: the TransE model of the best trial.
        """
        # construct the KG for training
        _, stmt = subsampling(self.dataset)
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
        trans_e.train(model_kwargs=dict(
            dim=best_t.params['model.embedding_dim'],
            scoring_fct_norm=best_t.params['model.scoring_fct_norm'],
            epochs=best_t.params['training.num_epochs'],
            batch_size=best_t.params['training.batch_size'],
            loss_margin=best_t.params['loss.margin'],
            optimizer_lr=best_t.params['optimizer.lr'],
            num_negs_per_pos=best_t.params['negative_sampler.num_negs_per_pos'],
            seed=seed))
        return trans_e
