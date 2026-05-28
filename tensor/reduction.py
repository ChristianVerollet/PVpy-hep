

def tensor_reduce_2point(integral, p, m1, m2):
    num = integral.numerator

    rank = detect_rank(num)

    if rank == 0:
        return B0(p2, m1, m2)

    elif rank == 1:
        mu = extract_index(num)
        return Momentum(mu) * B1(p2, m1, m2)

    elif rank == 2:
        mu, nu = extract_indices(num)
        return (
            Metric(mu, nu) * B00(p2, m1, m2)
            + Momentum(mu)*Momentum(nu) * B11(p2, m1, m2)
        )