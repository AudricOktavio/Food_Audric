from pydantic import BaseModel
from datetime import datetime

class TransactionCreate(BaseModel):
    trx_date: str
    trx_date_detail: str
    sales_id: str
    concept: str
    brand: str
    outlet: str
    district: str
    city: str
    menu_id: str
    menu_type: str
    menu_category: str
    menu_category_detail: str
    menu_name: str
    quantity: int
    user_id: str
    user_created_at: str
    user_tier_level: str
    user_gender: str