#! /usr/bin/env python

# from https://softwareengineering.stackexchange.com/questions/460573/coupon-redemption-system

import os
from uuid import UUID as GUID
from uuid import uuid3

import numpy as np
from beartype import beartype
from sqlalchemy import text
from sqlalchemy.orm import Session
from tqdm import tqdm
from valkey import Valkey

from so.so_2025_12.coupon.model import Base, Card, DbMgr, Device, Offer, get_session

COUPON_ENVIRONMENT = os.environ.get("COUPON_ENVIRONMENT", "Test")
assert COUPON_ENVIRONMENT in {"Prod", "Test"}
IN_PRODUCTION = COUPON_ENVIRONMENT == "Prod"

TOTAL_XACTS = 1_000_000 if IN_PRODUCTION else 10_000
TOTAL_OFFERS = TOTAL_XACTS // 1_000
TOTAL_CARDS = 2 * TOTAL_OFFERS
TOTAL_DEVICES = 4 * TOTAL_OFFERS

AMOUNT = 1.0  # dollar amount that each coupon redemption is worth


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


def _create_tables() -> None:
    with get_session() as sess:
        sess.execute(text("CREATE SCHEMA  IF NOT EXISTS  coupon"))
    Base.metadata.create_all(DbMgr.get_engine())


@beartype
class World:
    def __init__(self, amount: float = AMOUNT) -> None:

        self.offers = gen_population(TOTAL_OFFERS)
        self.cards = gen_population(TOTAL_CARDS)
        self.devices = gen_population(TOTAL_DEVICES)

        _create_tables()
        with get_session() as sess:

            for entity_cls, guids in [
                (Offer, self.offers),
                (Card, self.cards),
                (Device, self.devices),
            ]:
                sess.query(entity_cls).delete()

                for guid in sorted(set(guids)):
                    balance = int(0.3698 * amount * len(guids))
                    sess.add(entity_cls(guid=guid, balance=balance))

    def _decrement(
        self,
        sess: Session,
        tbl: type[Offer] | type[Card] | type[Device],
        guid: GUID,
    ) -> None:
        sess.execute(
            tbl.__table__.update().where(tbl.guid == guid).values(balance=tbl.balance - AMOUNT),
        )

    def redeem_coupons(self) -> None:
        with (
            get_session() as sess,
            Valkey() as client,
        ):
            gen = zip(self.offers, self.cards, self.devices, strict=True)
            for o_id, c_id, d_id in tqdm(gen, total=len(self.offers)):
                offer = sess.query(Offer).filter_by(guid=o_id).first()
                card = sess.query(Card).filter_by(guid=c_id).first()
                device = sess.query(Device).filter_by(guid=d_id).first()
                assert offer
                assert card
                assert device
                o_bal = float(f"{offer.balance}")
                c_bal = float(f"{card.balance}")
                d_bal = float(f"{device.balance}")

                if o_bal > AMOUNT and c_bal > AMOUNT and d_bal > AMOUNT:
                    self._decrement(sess, Offer, o_id)
                    self._decrement(sess, Card, c_id)
                    self._decrement(sess, Device, d_id)
                    client.incr("coupon_num_redemptions")
                else:
                    print(offer.guid.hex, card.guid.hex, device.guid.hex)

            sess.commit()


def main(*, verbose: bool = True) -> None:
    w = World()
    if verbose:
        w.redeem_coupons()


if __name__ == "__main__":
    main()
