from enum import unique
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import Column,String,Integer,Float,ForeignKey,DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import pymysql


Base = declarative_base()

# Creating the users table
    
class Records(Base):
    __tablename__ ='records'
    id = Column(Integer, primary_key=True)
    file_name = Column(String(50),nullable=True)
    filepath = Column(String(255))
    prediction = Column(String(255))
    created_at = Column(DateTime,default=datetime.utcnow, nullable=False)


    def __repr__(self) -> str:
        return f"{self.id} > {self.filepath}"

if __name__ == "__main__":
   
    engine = create_engine("mysql+pymysql://root:@12345@localhost/project?charset=utf8mb4")
    Base.metadata.create_all(engine)
  
