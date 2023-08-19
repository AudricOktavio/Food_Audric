from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./food_app.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

"""
This is my note for deployment case using mysql

from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy
import os

# Create the database engine
def connect_unix_socket() -> sqlalchemy.engine.base.Engine:
    db_user = "root"  # e.g. 'my-database-user'
    db_pass = "ayamgoreng"  # e.g. 'my-database-password'
    db_name = "crayadatab"  # e.g. 'my-database'
    unix_socket_path = "/cloudsql/harirayacloud:asia-southeast2:sqlapp"  # e.g. '/cloudsql/project:region:instance'

    pool = sqlalchemy.create_engine(
        # Equivalent URL:
        # mysql+pymysql://<db_user>:<db_pass>@/?unix_socket=<socket_path>/<cloud_sql_instance_name>
        sqlalchemy.engine.url.URL.create(
            drivername="mysql+pymysql",
            username=db_user,
            password=db_pass,
            database=db_name,
            query={"unix_socket": unix_socket_path},
        )
    )
    return pool

engine = connect_unix_socket()

# Create a SessionLocal class for creating database sessions
SessionLocal = sqlalchemy.orm.sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for declarative models
Base = sqlalchemy.ext.declarative.declarative_base()




from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLALCHEMY_DATABASE_URL = "postgresql://sqlapp:ayamgoreng@postgresserver/db"
# SQLALCHEMY_DATABASE_URL = "mysql+mysqlconnector://sqlapp:ayamgoreng@/crayadatab?unix_socket=/cloudsql/harirayacloud:asia-southeast2:sqlapp"
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:ayamgoreng@34.128.116.31/crayadatab"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
"""