import numpy as np

from SPARQLWrapper import SPARQLWrapper, JSON

from kgrec.datasets import Dataset

_load_sparql_limit = 10000

_query = """
SELECT ?rB ?p ?co ?ci ?cio ?cii WHERE { 
    { 
        SELECT DISTINCT ?rB ?p WHERE { 
            { 
                { 
                    ?rA ?p ?rB . 
                } UNION {
                    ?rB ?p ?rA .
                } UNION { 
                    ?rA ?p _:u .
                    ?rB ?p _:u . 
                } UNION { 
                    _:v ?p ?rA .
                    _:v ?p ?rB . 
                }
            FILTER ( ( isIRI( ?rB ) && !( ?rA = ?rB ) ) ) 
            } 
        }
    }
    OPTIONAL {
        ?rA ?p ?rB .
        {
            SELECT ?p ( COUNT( ?o1 ) AS ?co ) WHERE {
                ?rA ?p ?o1 .
                FILTER ( isIRI( ?o1 )) 
            } GROUP BY ?p
        } 
    }
    OPTIONAL { 
        {
            SELECT ?rB ?p ( COUNT( ?o2 ) AS ?ci ) WHERE {
                ?rB ?p ?rA ;
                    ?p ?o2 .
                FILTER ( isIRI( ?o2 ) && ?rB != ?rA) 
            } GROUP BY ?rB ?p
        } 
    }
    OPTIONAL {
            ?rA ?p _:a .
            ?rB ?p _:a .
        { 
            SELECT ?p ( COUNT( ?o1 ) AS ?cio ) WHERE {
                ?rA ?p _:x .
                ?o1 ?p _:x .
                FILTER ( isIRI( ?o1 )) 
            } GROUP BY ?p
        } 
    }
    OPTIONAL {
        _:b ?p ?rA .
        _:b ?p ?rB . 
        { 
            SELECT ?p ( COUNT( ?o2 ) AS ?cii ) WHERE {
                _:y ?p ?rA .
                _:y ?p ?o2 .
                FILTER ( isIRI( ?o2 )) 
            } GROUP BY ?p
        } 
    } 
}
ORDER BY ?rB ?p
OFFSET %%offset%%
LIMIT %%limit%%
"""


def query_for_ldsd(dataset: Dataset, r_a: str):
    sparql = SPARQLWrapper(
        endpoint=dataset.sparql_endpoint + '/query',
        defaultGraph=dataset.default_graph,
    )
    sparql.setReturnFormat(JSON)

    q = _query.replace('?rA', '<%s>' % r_a, -1)

    offset = 0
    values = {}
    while True:
        sparql.setQuery(q.replace('%%offset%%', str(offset), 1)
                        .replace('%%limit%%', str(_load_sparql_limit), 1))
        ret = sparql.queryAndConvert()

        n = 0
        for r in ret["results"]["bindings"]:
            r_b = r['rB']['value']
            if r_b not in values:
                values[r_b] = {}
            p = r['p']['value']
            di = np.float64(r['ci']['value']) if 'ci' in r else None
            do = np.float64(r['co']['value']) if 'co' in r else None
            dio = np.float64(r['cio']['value']) if 'cio' in r else None
            dii = np.float64(r['cii']['value']) if 'cii' in r else None
            values[r_b][p] = {
                'di': di,
                'do': do,
                'dio': dio,
                'dii': dii,
            }
            n += 1

        if n == _load_sparql_limit:
            offset += _load_sparql_limit
        else:
            break

    return values
