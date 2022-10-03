import progressbar as pb

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


def _count_statements(sparql: SPARQLWrapper) -> int:
    sparql.setQuery(_statements_count_query)
    ret = sparql.queryAndConvert()
    if len(ret['results']['bindings']) == 0:
        raise ValueError('couldn\'t fetch statements count')
    return int(ret['results']['bindings'][0]['cnt']['value'])


def collect_statements(dataset: Dataset) -> [[str, str, str]]:
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
    with pb.ProgressBar(max_value=_count_statements(sparql)) as p:
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

    return values
