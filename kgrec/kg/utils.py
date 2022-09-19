def create_ns_filter(ignore_ns: [str]) -> str:
    if ignore_ns:
        if len(ignore_ns) > 0:
            filter_str = '?g != <%s>' % ignore_ns[0]
            for ns in ignore_ns[1:]:
                filter_str += '&& ?g != <%s>' % ns
            return 'FILTER (%s)' % filter_str
    return ''
