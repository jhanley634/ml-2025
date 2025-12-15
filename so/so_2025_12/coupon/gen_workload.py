#! /usr/bin/env python

# from https://softwareengineering.stackexchange.com/questions/460573/coupon-redemption-system

import os
from uuid import UUID as GUID
from uuid import uuid3

import numpy as np

from so.so_2025_12.coupon.model import Base, Card, DbMgr, Device, Offer, get_session

COUPON_ENVIRONMENT = os.environ.get("COUPON_ENVIRONMENT", "Test")
assert COUPON_ENVIRONMENT in {"Prod", "Test"}
IN_PRODUCTION = COUPON_ENVIRONMENT == "Prod"

TOTAL_XACTS = 1_000_000 if IN_PRODUCTION else 10_000
TOTAL_OFFERS = TOTAL_XACTS // 1_000
TOTAL_CARDS = 2 * TOTAL_OFFERS
TOTAL_DEVICES = 4 * TOTAL_OFFERS

rng = np.random.default_rng(seed=0)
namespace = GUID(int=0)
guid_counter = [0]


def uuid() -> GUID:
    """Calls will return a sequence of things that look like guids,
    but which will be consistently reproducible across runs.
    """
    guid_counter[0] += 1
    return uuid3(namespace, f"{guid_counter[0]}")


def get_zipfian(n: int, alpha: float = 1.1) -> list[int]:
    samples = rng.zipf(alpha, size=int(TOTAL_XACTS * 11)) - 1
    s = np.array(list(filter(lambda x: x < n, samples))[:TOTAL_XACTS])
    assert len(s) == TOTAL_XACTS
    return list(map(int, s))


def gen_guids(n: int) -> list[GUID]:
    return [uuid() for _ in range(n)]


def gen_population(n: int) -> list[GUID]:
    entities = gen_guids(n)
    return [entities[i] for i in get_zipfian(n)]


def main(*, verbose: bool = False) -> None:
    engine = DbMgr.get_engine()
    Base.metadata.create_all(engine)

    offers = gen_population(TOTAL_OFFERS)
    cards = gen_population(TOTAL_CARDS)
    devices = gen_population(TOTAL_DEVICES)

    with get_session() as sess:

        for entity_cls, guids in [
            (Offer, offers),
            (Card, cards),
            (Device, devices),
        ]:
            sess.query(entity_cls).delete()

            for guid in sorted(set(guids)):
                sess.add(entity_cls(guid=guid))

    for triple in zip(offers, cards, devices, strict=True):
        if verbose:
            print(" ".join(guid.hex for guid in triple))


if __name__ == "__main__":
    main()
