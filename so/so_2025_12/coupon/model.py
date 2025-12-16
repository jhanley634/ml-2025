from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import DECIMAL, UUID, Column, Engine, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.orm.decl_api import DeclarativeMeta

TEMP = Path("/tmp")
COUPON_DB = TEMP / "coupon.db"

_coupon_schema = {"schema": "coupon"}


class DbMgr:
    _engine: Engine | None = None  # singleton

    @classmethod
    def get_engine(cls) -> Engine:
        if cls._engine is None:
            cls._engine = create_engine("postgresql://localhost/postgres")
        return cls._engine


@contextmanager
def get_session() -> Generator[Session]:
    with sessionmaker(bind=DbMgr.get_engine())() as sess:
        try:
            yield sess
        finally:
            sess.commit()


Base: DeclarativeMeta = declarative_base()


class Offer(Base):
    __tablename__ = "offer"
    __table_args__ = _coupon_schema

    guid = Column(UUID, primary_key=True)
    balance = Column(DECIMAL(10, 2), nullable=False)


class Card(Base):
    __tablename__ = "card"
    __table_args__ = _coupon_schema

    guid = Column(UUID, primary_key=True)
    balance = Column(DECIMAL(10, 2), nullable=False)


class Device(Base):
    __tablename__ = "device"
    __table_args__ = _coupon_schema

    guid = Column(UUID, primary_key=True)
    balance = Column(DECIMAL(10, 2), nullable=False)
