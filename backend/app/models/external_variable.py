from sqlalchemy import Column, Integer, String, Date, Float
from app.core.database import Base

class ExternalVariable(Base):
    __tablename__ = "external_variables"

    id = Column(Integer, primary_key=True, index=True)
    variable_date = Column(Date, nullable=False, index=True)
    variable_type = Column(String, nullable=False, index=True)
    variable_value = Column(Float, nullable=False)
    description = Column(String, nullable=True)