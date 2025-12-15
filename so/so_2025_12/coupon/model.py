from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import UUID, Column, Engine, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.orm.decl_api import DeclarativeMeta

TEMP = Path("/tmp")
COUPON_DB = TEMP / "coupon.db"


class DbMgr:
    _engine: Engine | None = None  # singleton

    @classmethod
    def get_engine(cls, db_file: Path = COUPON_DB) -> Engine:
        if cls._engine is None:
            cls._engine = create_engine(f"postgresql://localhost/postgres")
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

    guid = Column(UUID, primary_key=True)


class Card(Base):
    __tablename__ = "card"

    guid = Column(UUID, primary_key=True)


class Device(Base):
    __tablename__ = "device"

    guid = Column(UUID, primary_key=True)
