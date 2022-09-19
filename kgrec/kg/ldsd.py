import numpy as np

from SPARQLWrapper import SPARQLWrapper, JSON

from kgrec.datasets import Dataset

_load_sparql_limit = 500

_query = """
SELECT ?rB ?p ?do ?di ?dio ?dii WHERE
{
  {
    SELECT ?rB ?p WHERE
    {
      {?rA ?p ?rB}
      UNION
      {?rB ?p ?rA}
      UNION
      {
        ?rA ?p _:x .
        ?rB ?p _:x .
      }
      UNION
      {
        _:y ?p ?rA .
        _:y ?p ?rA .
      }
    }
    GROUP BY ?rB ?p
  }
  OPTIONAL {
    SELECT ?p (count(*) as ?do) WHERE
    {
      ?rA ?p ?o1 .
      FILTER (isIRI(?o1)) .
    }
    GROUP BY ?rB ?p
  }
  OPTIONAL {
    SELECT ?rB ?p (count(*) as ?di) WHERE
    {
      ?rB ?p ?rA .
      ?rB ?p ?o2 .
      FILTER (isIRI(?o2) && ?o2 != ?rA) .
    }
    GROUP BY ?rB ?p
  }
  OPTIONAL {
    SELECT ?rB ?p (count(*) as ?dio) WHERE
    {
      ?rA ?p ?o3 .
      ?rB ?p ?o3.
      FILTER (isIRI(?o3)) .
    }
    GROUP BY ?rB ?p
  }
  OPTIONAL {
    SELECT ?rB ?p (count(*) as ?dii) WHERE
    {
      ?o4 ?p ?rA .
      ?o4 ?p ?rB .
      FILTER (isIRI(?o4)) .
    }
    GROUP BY ?rB ?p
  }
  FILTER (isIRI(?rB) && ?rA != ?rB) .
}
ORDER BY ASC(?rB)
OFFSET %d
LIMIT %d
"""


def query_for_ldsd(dataset: Dataset, r_a: str):
    sparql = SPARQLWrapper(
        endpoint=dataset.sparql_endpoint,
        defaultGraph=dataset.default_graph,
    )
    sparql.setReturnFormat(JSON)

    q = _query.replace('?rA', '<%s>' % r_a, -1)

    offset = 0
    values = {}
    while True:
        sparql.setQuery(q % (offset, _load_sparql_limit))
        ret = sparql.queryAndConvert()

        n = 0
        for r in ret["results"]["bindings"]:
            r_b = r['rB']['value']
            if r_b not in values:
                values[r_b] = {}
            p = r['p']['value']
            di = np.float64(r['di']['value']) if 'di' in r else None
            do = np.float64(r['do']['value']) if 'do' in r else None
            dio = np.float64(r['dio']['value']) if 'dio' in r else None
            dii = np.float64(r['dii']['value']) if 'dii' in r else None
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

