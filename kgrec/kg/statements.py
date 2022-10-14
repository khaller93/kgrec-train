import numpy as np
import pandas as pd
import progressbar as pb

from os.path import join, exists
from os import makedirs

import requests
from SPARQLWrapper import SPARQLWrapper, JSON

from kgrec.datasets import Dataset

_load_sparql_limit = 10000

_statements_blank_node_sparql = """
SELECT ?s ?p ?o WHERE {
    ?s ?p ?o
    FILTER ((isBlank(?s) || isBlank(?o)) && (isIRI(?o) || isBlank(?o)))
}
"""

_statements_count_query = """
SELECT (count(*) as ?cnt) WHERE {
    ?s ?p ?o
    FILTER (isIRI(?s) && isIRI(?o))
}
"""

_statements_iri_node_query = """
SELECT ?s ?p ?o WHERE {
    ?s ?p ?o
    FILTER (isIRI(?s) && isIRI(?o))
}
ORDER BY ASC(?s) ASC(?p) ASC(?o)
OFFSET %d
LIMIT %d
"""


def collect_statements(dataset: Dataset,
                       model_out_directory: str) -> np.ndarray:
    if dataset.statements_file:
        return __load_statements_from_file(dataset,
                                           model_out_directory)
    else:
        return __load_statements_from_triplestore(dataset)


def __load_statements_from_file(dataset: Dataset,
                                model_out_directory: str) -> np.ndarray:
    fp = join(model_out_directory, dataset.name.lower(),
              dataset.statements_file[0])

    if not exists(fp):
        makedirs(join(model_out_directory, dataset.name.lower()))
        response = requests.get(dataset.statements_file[1])
        with open(fp, 'wb+') as f:
            f.write(response.content)

    return pd.read_csv(fp, compression='gzip', header=None, sep='\t').values


def __load_statements_from_triplestore(dataset: Dataset) -> np.ndarray:
    sparql = SPARQLWrapper(
        endpoint=dataset.sparql_endpoint + '/query',
        defaultGraph=dataset.default_graph,
    )
    sparql.setReturnFormat(JSON)

    values = []

    print('collect statements with blank nodes ...')
    sparql.setQuery(_statements_blank_node_sparql)
    ret = sparql.queryAndConvert()
    for r in ret["results"]["bindings"]:
        values.append([r['s']['value'], r['p']['value'], r['o']['value']])

    print('collect statements ...')
    offset = 0
    print('get statement count ...')
    with pb.ProgressBar(max_value=__count_statements(sparql)) as p:
        print('fetch statements ...')
        while True:
            sparql.setQuery(_statements_iri_node_query % (offset,
                                                          _load_sparql_limit))
            n = 0
            ret = sparql.queryAndConvert()
            for r in ret["results"]["bindings"]:
                n += 1
                p.update(n)
                values.append(
                    [r['s']['value'], r['p']['value'], r['o']['value']])

            if n == _load_sparql_limit:
                offset += _load_sparql_limit
            else:
                break

    return np.array(values)


def __count_statements(sparql: SPARQLWrapper) -> int:
    sparql.setQuery(_statements_count_query)
    ret = sparql.queryAndConvert()
    if len(ret['results']['bindings']) == 0:
        raise ValueError('couldn\'t fetch statements count')
    return int(ret['results']['bindings'][0]['cnt']['value'])
