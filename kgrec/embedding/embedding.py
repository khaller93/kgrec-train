import pandas as pd

from abc import abstractmethod
from kgrec.datasets import Dataset
from os import makedirs
from os.path import exists, join


class Embedding:
    """ a KG embedding that can be trained """

    def __init__(self, dataset: Dataset):
        """
        creates a new embedding for the specified dataset.

        :param dataset: dataset to which apply the embedding.
        """
        self.dataset = dataset
        self.model = None
        self.model_kwargs = None

    @abstractmethod
    def _get_name(self) -> str:
        """
        gets the name of this embedding.

        :return: the name of this embedding.
        """
        raise NotImplementedError('must be implemented')

    @abstractmethod
    def _perform_training(self, model_kwargs, training_kwargs) -> pd.DataFrame:
        """
        trains this embedding with the given arguments.

        :param model_kwargs: arguments for the training of this embedding.
        :param training_kwargs: arguments for the training process (e.g. number
        of threads to use)
        """
        raise NotImplementedError('must be implemented')

    def train(self, model_kwargs, training_kwargs=None):
        """
        trains this embedding with the given arguments.

        :param model_kwargs: arguments for the training of this embedding.
        :param training_kwargs: arguments for the training process (e.g. number
        of threads to use)
        """
        self.model = self._perform_training(model_kwargs, training_kwargs)
        self.model_kwargs = model_kwargs

    def _get_model_name(self):
        """
        gets the  model name for the last trained model.

        :return: the model name for the last trained model.
        """
        arg_names = [key for key, _ in self.model_kwargs.items()]
        arg_names.sort()
        # construct abbreviations
        arg_abbreviations = []
        for arg_name in arg_names:
            vs = str.split(arg_name, '_')
            abr = ''.join([s[:1] for s in vs])
            n = 1
            while abr in arg_abbreviations and n < len(arg_name):
                n += 1
                abr = ''.join([s[:n] for s in vs])
            arg_abbreviations.append(abr)
        # construct model name
        suffix = []
        for i, abr in enumerate(arg_abbreviations):
            suffix.append('%s%s' % (abr, str(self.model_kwargs[arg_names[i]])))
        return ('%s_%s' % (self._get_name(), '_'.join(suffix))) \
            .replace('.', '_')

    def write_to(self, model_dir_path: str):
        """
        writes the trained model to the specified directory. The train method
        must be called and successfully run before calling this method.

        :param model_dir_path: to which the model shall be written.
        """
        if self.model is None:
            raise RuntimeError(
                'you must train the model before calling this method')
        out_dir = join(model_dir_path, self.dataset.name)
        if not exists(out_dir):
            makedirs(out_dir)
        index_f = join(out_dir, 'entities.tsv.gz')
        if not exists(index_f):
            df = self.dataset.relevant_entities.copy()
            del df['key_index']
            df.to_csv(index_f, header=False, sep='\t', compression='gzip')
        self.model.to_csv(join(out_dir, '%s.tsv.gz' % self._get_model_name()),
                          header=False, sep='\t', compression='gzip')
