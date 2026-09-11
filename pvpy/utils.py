
def pv_symbol_key(s):
    name = str(s)

    # put mu last if you ever include it
    if name == "mu" or name == "μ":
        return (99, name)

    # invariants first: p2, s, t, u, ...
    if name.startswith("p"):
        return (0, name)

    # masses: m, m1, m2, ...
    if name.startswith("m"):
        return (1, name)

    # everything else
    return (10, name)