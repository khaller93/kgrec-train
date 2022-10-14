import numpy as np
import pandas as pd
import os.path as path
import progressbar as pb

from collections.abc import Mapping, Sequence

from kgrec.datasets import Dataset
from kgrec.kg.entities import get_entities
from kgrec.kg.ldsd import query_for_ldsd


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
        if val['do'] is not None and val['do'] != 0:
            do_sum += np.float64(1) / (np.float64(1) + np.log(val['do']))
        if val['di'] is not None and val['di'] != 0:
            di_sum += np.float64(1) / (np.float64(1) + np.log(val['di']))
        if val['dio'] is not None and val['dio'] != 0:
            dio_sum += np.float64(1) / (np.float64(1) + np.log(val['dio']))
        if val['dii'] is not None and val['dii'] != 0:
            dii_sum += np.float64(1) / (np.float64(1) + np.log(val['dii']))
    return np.float64(1) / (np.float64(1) + do_sum + di_sum + dio_sum + dii_sum)


def _process_query_response(response: Mapping[str, Mapping[str, dict]]):
    val_list = []
    for r_b, properties in response.items():
        val_list.append((r_b, _compute_result(properties)))
    return val_list


def train(dataset: Dataset, model_out_directory: str):
    ent = get_entities(dataset, model_out_directory)['iri'].values.tolist()
    ent_id = _get_entity_id_map(ent)

    model_file = path.join(model_out_directory, dataset.name.lower(),
                           'ldsd.tsv')
    with open(model_file, 'w') as f:
        with pb.ProgressBar(max_value=len(ent)) as p:
            for i, r_a in enumerate(ent):
                f.write('%d,%d\t%s\n' % (i, i, np.float(0)))
                resp = query_for_ldsd(dataset, r_a)
                for r_b, val in _process_query_response(resp):
                    print('(%s,%s) = %s' % (r_a, r_b, val))
                    if r_b not in ent_id:
                        continue
                    j = ent_id[r_b]
                    f.write('%d,%d\t%s\n' % (i, j, val))
                p.update(i)

