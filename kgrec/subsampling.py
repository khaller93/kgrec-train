import pandas as pd

from kgrec.datasets import Dataset
from typing import Callable, Tuple


SubSamplingFunc = Callable[[Dataset], Tuple[pd.DataFrame, pd.DataFrame]]


def id_sub(dataset: Dataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    a subsampling function that just returns the complete KG.

    :return: the complete KG.
    """
    return dataset.relevant_entities, dataset.statements


def random_fraction_sub(frac: float, seed: int) -> SubSamplingFunc:
    """
    a subsampling function that returns a fraction of the original KG.

    :param frac: fraction of the KG that shall remain.
    :param seed: random seed for the selection of entities.
    :return: a subsampling function that returns fraction of KG.
    """
    if frac <= 0.0 or frac > 1.0:
        raise ValueError('subsampling value is out of range (]0,1])')
    elif frac == 1.0:
        return id_sub
    else:
        def func(dataset: Dataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
            ent_df = dataset.index
            ent_sample = ent_df.sample(frac=frac, random_state=seed)
            ent_sample_set = set(ent_sample.index)
            stmt_df = dataset.statements.copy()
            stmt_df = stmt_df[stmt_df['subj'].isin(ent_sample_set)]
            stmt_df = stmt_df[stmt_df['obj'].isin(ent_sample_set)]
            return ent_sample, stmt_df

        return func
