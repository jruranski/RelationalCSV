from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./dataloom.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    original_filename = Column(String)
    file_path = Column(String)
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)
    file_size = Column(Integer)
    row_count = Column(Integer)
    columns = Column(String)

# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to execute SQL directly
def execute_sql(sql_query):
    with engine.connect() as connection:
        result = connection.execute(sql_query)
        return result

# Function to retrieve database schema information
def get_db_schema():
    """Get all tables and their structure from the database"""
    inspector = inspect(engine)
    schema_info = []
    
    # Get all table names
    table_names = inspector.get_table_names()
    
    for table_name in table_names:
        if table_name == 'uploaded_files':
            # Skip our internal table
            continue
            
        columns = []
        for column in inspector.get_columns(table_name):
            columns.append({
                'name': column['name'],
                'type': str(column['type']),
                'nullable': column['nullable']
            })
            
        primary_keys = inspector.get_pk_constraint(table_name).get('constrained_columns', [])
        foreign_keys = []
        for fk in inspector.get_foreign_keys(table_name):
            foreign_keys.append({
                'column': fk['constrained_columns'],
                'referred_table': fk['referred_table'],
                'referred_columns': fk['referred_columns']
            })
            
        indexes = []
        for idx in inspector.get_indexes(table_name):
            indexes.append({
                'name': idx['name'],
                'columns': idx['column_names'],
                'unique': idx['unique']
            })
            
        table_info = {
            'name': table_name,
            'columns': columns,
            'primary_keys': primary_keys,
            'foreign_keys': foreign_keys,
            'indexes': indexes
        }
        
        schema_info.append(table_info)
        
    return schema_info 