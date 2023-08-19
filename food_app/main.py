from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from . import crud, prediction, models, schemas
from .database import SessionLocal, engine

from starlette.requests import Request
from starlette.responses import JSONResponse
from pydantic import EmailStr, BaseModel
from typing import List

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency - Databse get
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# FastAPI endpoint to CRUD for datas we have on SQLite, used as a simulation for real time sales and transaction and how we utilize our model
# Please note that I could also use MySQL, Postgre or NoSQL for this project however SQLite is most effective to do as of now

@app.post("/transactions/") #create transaction
def create_transaction(transaction: schemas.TransactionCreate, db: Session = Depends(get_db)):
    return crud.create_transaction(db, transaction)

@app.get("/transactions/{sales_id}") #get transaction
def read_transaction(sales_id: str, db: Session = Depends(get_db)):
    db_transaction = crud.get_transaction(db, sales_id)
    if db_transaction is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return db_transaction

@app.get("/transactions/") # get transactions
def read_transactions(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    transactions = crud.get_transactions(db, skip=skip, limit=limit)
    return transactions

@app.delete("/transactions/{sales_id}") # delete transaction
def delete_transaction(sales_id: str, db: Session = Depends(get_db)):
    deleted_transaction = crud.delete_transaction(db, sales_id)
    if deleted_transaction is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return deleted_transaction

@app.get("/segmentation/{user_id}") #ML Segmentation example
def read_segmentation(user_id: str, n: int, db: Session = Depends(get_db)):
    user_segmentation = prediction.combine_behaviors(user_id, n)
    return user_segmentation

@app.get("/recommendation/{user_id}") #ML Recommendation example
def get_recommendation(user_id: str, n: int, db: Session = Depends(get_db)):
    recommendation = prediction.predict_menu(user_id, n)
    return recommendation

# It is also possible to create an API to specifically retrain my DL
# Recommendation Model