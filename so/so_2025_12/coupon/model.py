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
            cls._engine = create_engine(f"sqlite:///{db_file}")
        return cls._engine


@contextmanager
def get_session() -> Generator[Session]:
    with sessionmaker(bind=DbMgr.get_engine())() as sess:
        try:
            yield sess
        finally:
            sess.commit()


Base = declarative_base()
assert isinstance(Base, DeclarativeMeta)


class Offer(Base):  # type: ignore
    __tablename__ = "offer"

    guid = Column(UUID, primary_key=True)
