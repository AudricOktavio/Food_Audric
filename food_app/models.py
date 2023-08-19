from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Date
from sqlalchemy.orm import relationship

from .database import Base

# SQLAlchemy model for database interaction
class Transaction(Base):
    __tablename__ = "transactions"
    trx_date = Column(Date)
    trx_date_detail = Column(String)
    sales_id = Column(String, primary_key=True, index=True)
    concept = Column(String)
    brand = Column(String)
    outlet = Column(String)
    district = Column(String)
    city = Column(String)
    menu_id = Column(String)
    menu_type = Column(String)
    menu_category = Column(String)
    menu_category_detail = Column(String)
    menu_name = Column(String)
    quantity = Column(Integer)
    user_id = Column(String)
    user_created_at = Column(String)
    user_tier_level = Column(String)
    user_gender = Column(String)