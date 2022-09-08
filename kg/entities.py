import os.path as path
import pandas as pd

from os import mkdir
from SPARQLWrapper import SPARQLWrapper, JSON

from datasets import Dataset

_load_sparql_limit = 500

_entities_sparql_query = """
SELECT DISTINCT ?s WHERE {
    {?s ?p ?o} UNION {?u ?b ?s}
    FILTER(isIRI(?s))
}
ORDER BY ASC(?s)
OFFSET %d
LIMIT %d
"""


def gather_entities_from_sparql_endpoint(sparql_endpoint: str) -> pd.DataFrame:
    sparql = SPARQLWrapper(
        sparql_endpoint
    )
    sparql.setReturnFormat(JSON)

    not_all = True
    offset = 0
    values = []
    while not_all:
        sparql.setQuery(_entities_sparql_query % (offset, _load_sparql_limit))
        ret = sparql.queryAndConvert()

        n = 0
        for r in ret["results"]["bindings"]:
            values.append(r['s']['value'])
            n += 1

        if n == _load_sparql_limit:
            offset += _load_sparql_limit
        else:
            not_all = False
    return pd.DataFrame(values, columns=['iri'])


def _write_entities_to_file(entities: pd.DataFrame, entities_file: str):
    d = path.dirname(entities_file)
    if not path.exists(d) or not path.isdir(d):
        mkdir(d)
    entities.to_csv(entities_file, sep='\t')


def read_entities_from_file(entities_file: str) -> pd.DataFrame:
    df = pd.read_csv(entities_file, sep="\t", header=0, names=['index', 'iri'])
    return df


def get_entities(dataset: Dataset, model_out_dir: str) -> pd.DataFrame:
    entities_file_path = path.join(model_out_dir, dataset.name.lower(),
                                   'entities.tsv')
    if path.exists(entities_file_path) and path.isfile(entities_file_path):
        return read_entities_from_file(entities_file_path)
    else:
        entities = gather_entities_from_sparql_endpoint(dataset.sparql_endpoint)
        _write_entities_to_file(entities, entities_file_path)
        return entities
