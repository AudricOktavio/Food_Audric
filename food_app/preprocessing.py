from sqlalchemy.orm import Session
from . import models, schemas, preprocessing
import pandas as pd
from datetime import datetime
import numpy as np
from typing import Dict, Text

#this notebook will mostly be used for preprocessing data

def convert_db_to_dataframe(db: Session):
  # 'Transaction' is database model
  transactions = db.query(models.Transaction).all()

  # Convert the list of Transaction objects to a dictionary
  transaction_data = [transaction.__dict__ for transaction in transactions]

  # Convert the dictionary to a Pandas DataFrame
  df = pd.DataFrame(transaction_data)

  return df

def extract_hour(datetime_obj):
    return datetime_obj.hour

def extract_minutes(datetime_obj):
    return datetime_obj.minute

def extract_day(datetime_obj):
    return datetime_obj.day

def extract_month(datetime_obj):
    return datetime_obj.month

def extract_year(datetime_obj):
    return datetime_obj.year

def preprocessing_data(db: Session):
    df = convert_db_to_dataframe(db)
    df['trx_date_detail'] = pd.to_datetime(df['trx_date_detail'], format="%Y-%m-%d %H:%M:%S %Z")
    df['user_created_at'] = pd.to_datetime(df['user_created_at'])
    df['hour'] = df['trx_date_detail'].apply(extract_hour)
    df['day'] = df['trx_date_detail'].apply(extract_day)
    df['month'] = df['trx_date_detail'].apply(extract_month)
    df['year'] = df['trx_date_detail'].apply(extract_year)
    df['user_created_year'] = df['user_created_at'].dt.year
    df['user_created_month'] = df['user_created_at'].dt.month
    df['user_created_day'] = df['user_created_at'].dt.day
    df['user_created_hour'] = df['user_created_at'].dt.hour
    df = df.drop('trx_date_detail', axis=1)
    df = df.drop('trx_date', axis=1)
    df = df.drop('user_created_at', axis=1)
    df['user_id'] = df['user_id'].astype(str)
    df['sales_id'] = df['sales_id'].astype(str)
    df['quantity'] = df['quantity'].astype(np.float32)
    return df

