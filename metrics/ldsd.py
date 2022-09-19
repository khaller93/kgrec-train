import numpy as np
import pandas as pd
import os.path as path
import progressbar

from collections.abc import Mapping, Sequence
from datasets import Dataset
from kg.entities import get_entities
from kg.ldsd import query_for_ldsd

widgets = [
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
]


def _get_entity_id_map(entities: [str]):
    m = {}
    for i, entity in enumerate(entities):
        m[entity] = i
    return m


def _compute_result(properties: Mapping[str, dict]) -> np.float:
    do_sum = np.float64(0)
    di_sum = np.float64(0)
    dio_sum = np.float64(0)
    dii_sum = np.float64(0)
    for val in properties.values():
        if val['do'] is not None:
            do_sum += np.float64(1) / (np.float64(1) + np.log(val['do']))
        if val['di'] is not None:
            di_sum += np.float64(1) / (np.float64(1) + np.log(val['di']))
        if val['dio'] is not None:
            dio_sum += np.float64(1) / (np.float64(1) + np.log(val['dio']))
        if val['dii'] is not None:
            dii_sum += np.float64(1) / (np.float64(1) + np.log(val['dii']))
    return np.float64(1) / (np.float64(1) + do_sum + di_sum + dio_sum + dii_sum)


def _process_query_response(response: Mapping[str, Mapping[str, dict]]) -> [
        (str, np.float)]:
    val_list = []
    for r_b, properties in response.items():
        val_list.append((r_b, _compute_result(properties)))
    return val_list


def train(dataset: Dataset, model_out_directory: str):
    ent = get_entities(dataset, model_out_directory)['iri'].values.tolist()
    ent_id = _get_entity_id_map(ent)

    sim_x = np.ones((len(ent), len(ent)), dtype=np.float64)
    with progressbar.ProgressBar(max_value=len(ent)) as p:
        for i, r_a in enumerate(ent):
            sim_x[i, i] = 0
            resp = query_for_ldsd(dataset, r_a)
            for r_b, val in _process_query_response(resp):
                j = ent_id[r_b]
                sim_x[i, j] = val
            p.update(i)

    model_file = path.join(model_out_directory, dataset.name.lower(),
                           'ldsd.tsv')
    pd.DataFrame(sim_x).to_csv(model_file, index=False, sep='\t', header=False)
