from SPARQLWrapper import SPARQLWrapper, JSON

from datasets import Dataset
from kg.utils import create_ns_filter

_load_sparql_limit = 500

_statements_blank_node_sparql = """
SELECT ?s ?p ?o WHERE {
    Graph ?g {
        ?s ?p ?o
        FILTER ((isBlank(?s) || isBlank(?o)) && (isIRI(?o) || isBlank(?o)))
    }
    %s
}
"""

_statements_iri_node_sparql = """
SELECT ?s ?p ?o WHERE {
    Graph ?g {
        ?s ?p ?o
        FILTER (isIRI(?s) && isIRI(?o))
    }
    %s
}
OFFSET %d
LIMIT %d
"""


def collect_statements(dataset: Dataset) -> [[str, str, str]]:
    sparql = SPARQLWrapper(
        endpoint=dataset.sparql_endpoint,
        defaultGraph=dataset.default_graph,
    )
    sparql.setReturnFormat(JSON)

    values = []
    ignore_ns = dataset.ignore_named_graphs

    sparql.setQuery(_statements_blank_node_sparql % create_ns_filter(ignore_ns))
    ret = sparql.queryAndConvert()
    for r in ret["results"]["bindings"]:
        values.append([r['s']['value'], r['p']['value'], r['o']['value']])

    offset = 0
    while True:
        sparql.setQuery(_statements_iri_node_sparql % (create_ns_filter(
            ignore_ns), offset, _load_sparql_limit))
        n = 0
        ret = sparql.queryAndConvert()
        for r in ret["results"]["bindings"]:
            n += 1
            values.append([r['s']['value'], r['p']['value'], r['o']['value']])

        if n == _load_sparql_limit:
            offset += _load_sparql_limit
        else:
            break

    return values
