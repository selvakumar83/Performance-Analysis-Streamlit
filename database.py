# models/database.py
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Result(Base):
    __tablename__ = 'results'
    id = Column(Integer, primary_key=True)
    algorithm = Column(String)
    accuracy = Column(Float)

engine = create_engine('sqlite:///../results.db', echo=True)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
