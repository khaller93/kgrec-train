import os.path as path
import pandas as pd
import progressbar as pb

from os import makedirs
from SPARQLWrapper import SPARQLWrapper, JSON

from kgrec.datasets import Dataset

_load_sparql_limit = 10000

_count_entities_query = """
SELECT (count(distinct ?s) as ?cnt) WHERE
{
    {?s ?p ?o} UNION {?u ?b ?s}
    FILTER(isIRI(?s))
}
"""

_entities_sparql_query = """
SELECT ?s WHERE
{
    SELECT distinct ?s WHERE {
        {?s ?p ?o} UNION {?u ?b ?s}
        FILTER(isIRI(?s))
    }
    ORDER BY ASC(?s)
}
OFFSET %%offset%%
LIMIT %%limit%%
"""


def _count_entities(sparql: SPARQLWrapper) -> int:
    sparql.setQuery(_count_entities_query)
    ret = sparql.queryAndConvert()
    if len(ret['results']['bindings']) == 0:
        raise ValueError('couldn\'t fetch entity count')
    return int(ret['results']['bindings'][0]['cnt']['value'])


def gather_entities_from_sparql_endpoint(dataset: Dataset) -> pd.DataFrame:
    sparql = SPARQLWrapper(
        endpoint=dataset.sparql_endpoint,
        defaultGraph=dataset.default_graph,
    )
    sparql.setReturnFormat(JSON)
    offset = 0
    values = []
    print('get entity count ...')
    with pb.ProgressBar(max_value=_count_entities(sparql)) as p:
        print('fetch entities ...')
        while True:
            sparql.setQuery(_entities_sparql_query
                            .replace('%%offset%%', str(offset), 1)
                            .replace('%%limit%%', str(_load_sparql_limit), 1))
            ret = sparql.queryAndConvert()

            if len(ret["results"]["bindings"]) == 0:
                break

            for r in ret["results"]["bindings"]:
                values.append(r['s']['value'])
                p.update()

            offset += _load_sparql_limit

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
