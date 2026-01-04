#! /usr/bin/env python

# from https://softwareengineering.stackexchange.com/questions/460573/coupon-redemption-system

import os
from decimal import Decimal
from itertools import count
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


rng = np.random.default_rng(seed=0)
namespace = GUID(int=0)
guid_counter = count(0)


def uuid() -> GUID:
    """Calls will return a sequence of things that look like guids,
    but which will be consistently reproducible across runs.
    """
    return uuid3(namespace, f"{next(guid_counter)}")


def gen_guids(n: int) -> list[GUID]:
    return [uuid() for _ in range(n)]


def get_zipfian(n: int, count: int, alpha: float = 1.1) -> list[int]:
    samples = rng.zipf(alpha, size=int(count * 11)) - 1
    s = np.array(list(filter(lambda x: x < n, samples))[:count])
    assert len(s) == count
    return list(map(int, s))


def gen_population(n: int, *, count: int = TOTAL_XACTS) -> list[GUID]:
    entities = gen_guids(n)
    return [entities[i] for i in get_zipfian(n, count)]


def _create_tables() -> None:
    with get_session() as sess:
        sess.execute(text("CREATE SCHEMA  IF NOT EXISTS  coupon"))
    Base.metadata.create_all(DbMgr.get_engine())


@beartype
class World:

    AMOUNT = 1.0  # dollar amount that each coupon redemption is worth

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

        self.amount = amount

    def _decrement(
        self,
        sess: Session,
        tbl: type[Offer] | type[Card] | type[Device],
        guid: GUID,
    ) -> None:
        sess.execute(
            tbl.__table__.update()
            .where(tbl.guid == guid)
            .values(balance=tbl.balance - self.amount),
        )

    def redeem_coupons(self) -> None:
        def fetch_balance(
            tbl: type[Offer] | type[Card] | type[Device],
            id_: GUID,
        ) -> tuple[Decimal, str]:
            row = sess.query(tbl).filter_by(guid=id_).first()
            assert row
            return Decimal(f"{row.balance}"), f"{row.guid.hex}"

        with (
            get_session() as sess,
            Valkey() as client,
        ):
            gen = zip(self.offers, self.cards, self.devices, strict=True)
            for o_id, c_id, d_id in tqdm(gen, total=len(self.offers)):
                offer_b, offer_h = fetch_balance(Offer, o_id)
                card_b, card_h = fetch_balance(Card, c_id)
                device_b, device_h = fetch_balance(Device, d_id)
                amt = Decimal(f"{self.amount}")

                if offer_b > amt and card_b > amt and device_b > amt:
                    self._decrement(sess, Offer, o_id)
                    self._decrement(sess, Card, c_id)
                    self._decrement(sess, Device, d_id)
                    client.incr("coupon_num_redemptions")
                else:
                    print(offer_h, card_h, device_h)

            sess.commit()


def main(*, verbose: bool = True) -> None:
    w = World()
    if verbose:
        w.redeem_coupons()


if __name__ == "__main__":
    main()
