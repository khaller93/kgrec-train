import os.path as path
import pandas as pd

from os import makedirs
from SPARQLWrapper import SPARQLWrapper, JSON

from datasets import Dataset

_load_sparql_limit = 500

_entities_sparql_query = """
SELECT ?s WHERE {
    {
        SELECT DISTINCT ?s WHERE {
            {?s ?p ?o} UNION {?u ?b ?s}
            FILTER(isIRI(?s))   
        } ORDER BY ASC(?s)
    }
}
OFFSET %d
LIMIT %d
"""


def gather_entities_from_sparql_endpoint(dataset: Dataset) -> pd.DataFrame:
    sparql = SPARQLWrapper(
        endpoint=dataset.sparql_endpoint,
        defaultGraph=dataset.default_named_graph,
    )
    sparql.setReturnFormat(JSON)

    offset = 0
    values = []
    while True:
        sparql.setQuery(_entities_sparql_query % (offset, _load_sparql_limit))
        ret = sparql.queryAndConvert()

        n = 0
        for r in ret["results"]["bindings"]:
            values.append(r['s']['value'])
            n += 1

        if n == _load_sparql_limit:
            offset += _load_sparql_limit
        else:
            break

    return pd.DataFrame(values, columns=['iri'])


def _write_entities_to_file(entities: pd.DataFrame, entities_file: str):
    d = path.dirname(entities_file)
    if not path.exists(d) or not path.isdir(d):
        makedirs(d)
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
        entities = gather_entities_from_sparql_endpoint(dataset)
        _write_entities_to_file(entities, entities_file_path)
        return entities
