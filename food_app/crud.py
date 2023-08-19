from sqlalchemy.orm import Session
from datetime import datetime

import secrets
import string

from . import models, schemas
from uuid import uuid4

#Here we have the CRUD part to modify our data

def create_transaction(db: Session, transaction: schemas.TransactionCreate): #Creating new transaction
    trx_date = datetime.strptime(transaction.trx_date, "%Y-%m-%d").date()
    trx_date_detail = datetime.strptime(transaction.trx_date_detail, "%Y-%m-%d %H:%M:%S %Z")
    
    db_transaction = models.Transaction(
        trx_date=trx_date,
        trx_date_detail=trx_date_detail,
        sales_id=transaction.sales_id,
        concept=transaction.concept,
        brand=transaction.brand,
        outlet=transaction.outlet,
        district=transaction.district,
        city=transaction.city,
        menu_id=transaction.menu_id,
        menu_type=transaction.menu_type,
        menu_category=transaction.menu_category,
        menu_category_detail=transaction.menu_category_detail,
        menu_name=transaction.menu_name,
        quantity=transaction.quantity,
        user_id=transaction.user_id,
        user_created_at=transaction.user_created_at,
        user_tier_level=transaction.user_tier_level,
        user_gender=transaction.user_gender
    )
    db.add(db_transaction)
    db.commit()
    db.refresh(db_transaction)
    return db_transaction

def get_transaction(db: Session, transaction_id: str): #Getting specific transaction
    return db.query(models.Transaction).filter(models.Transaction.sales_id == transaction_id).all()

def get_transactions(db: Session, skip: int = 0, limit: int = 10): #Getting all transaction
    return db.query(models.Transaction).offset(skip).limit(limit).all()

def delete_transaction(db: Session, transaction_id: str): #Deleting a transaction
    db_transaction = db.query(models.Transaction).filter(models.Transaction.sales_id == transaction_id).first()
    if db_transaction:
        db.delete(db_transaction)
        db.commit()
        return db_transaction