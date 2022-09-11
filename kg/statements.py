from SPARQLWrapper import SPARQLWrapper, JSON

_load_sparql_limit = 500

_statements_blank_node_sparql = """
SELECT ?s ?p ?o WHERE {
    ?s ?p ?o
    FILTER ((isBlank(?s) || isBlank(?o)) && (isIRI(?o) || isBlank(?o)))
}
"""

_statements_iri_node_sparql = """
SELECT ?s ?p ?o WHERE {
    ?s ?p ?o
    FILTER (isIRI(?s) && isIRI(?o))
}
OFFSET %d
LIMIT %d
"""


def collect_statements(sparql_endpoint: str) -> [[str, str, str]]:
    sparql = SPARQLWrapper(
        sparql_endpoint
    )
    sparql.setReturnFormat(JSON)

    values = []

    sparql.setQuery(_statements_blank_node_sparql)
    ret = sparql.queryAndConvert()
    for r in ret["results"]["bindings"]:
        values.append([r['s']['value'], r['p']['value'], r['o']['value']])

    offset = 0
    while True:
        sparql.setQuery(_statements_iri_node_sparql % (offset,
                                                       _load_sparql_limit))
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
