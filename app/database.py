from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, inspect, text
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
        # Begin a transaction 
        trans = connection.begin()
        try:
            # Split the input SQL string into individual statements
            # Simple split by semicolon won't work well with complex SQL
            # that might have semicolons inside quotes or comments
            raw_statements = [stmt.strip() for stmt in sql_query.split(';') if stmt.strip()]
            statements = []
            
            # Basic cleaning of statements
            for stmt in raw_statements:
                # Skip empty statements
                if not stmt.strip():
                    continue
                
                # Basic validation
                if stmt.upper().startswith(('CREATE', 'DROP', 'ALTER', 'INSERT', 'UPDATE', 'DELETE', 'SELECT')):
                    # Statement looks valid, clean it further
                    # Remove trailing commas before closing parenthesis
                    cleaned_stmt = stmt.replace(",\n)", "\n)")
                    cleaned_stmt = cleaned_stmt.replace(", )", ")")
                    statements.append(cleaned_stmt)
                else:
                    print(f"Warning: Skipping invalid SQL statement: {stmt[:50]}...")
            
            results = []
            
            # Execute each statement separately
            for statement in statements:
                try:
                    print(f"Executing statement: {statement}")
                    # Use SQLAlchemy's text() to properly prepare the statement
                    result = connection.execute(text(statement))
                    results.append(result)
                except Exception as e:
                    # Print more info about the failed statement
                    print(f"Failed statement: {statement}")
                    print(f"Error: {str(e)}")
                    trans.rollback()
                    raise Exception(f"Error executing statement: {statement}\nError: {str(e)}")
            
            # Commit the transaction if all statements succeeded
            trans.commit()
            return results
        except Exception as e:
            trans.rollback()
            raise e

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

# Function to get all user tables (excluding system and internal tables)
def get_user_tables():
    """Get a list of all user-created tables in the database (excluding internal tables)"""
    inspector = inspect(engine)
    tables = []
    
    # Get all table names
    table_names = inspector.get_table_names()
    
    for table_name in table_names:
        # Skip internal tables
        if table_name == 'uploaded_files' or table_name.startswith('sqlite_'):
            continue
        
        # Get column count
        columns = inspector.get_columns(table_name)
        column_count = len(columns)
        
        # Try to get row count (might be slow for large tables)
        row_count = 0
        try:
            with engine.connect() as connection:
                result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = result.scalar()
        except:
            pass
        
        tables.append({
            "name": table_name,
            "column_count": column_count,
            "row_count": row_count
        })
    
    return tables

# Function to delete a table
def delete_table(table_name):
    """Delete a table from the database"""
    # Check if table exists
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return False, f"Table '{table_name}' not found"
    
    # Don't allow deleting internal tables
    if table_name == 'uploaded_files' or table_name.startswith('sqlite_'):
        return False, f"Cannot delete internal table '{table_name}'"
    
    try:
        with engine.connect() as connection:
            trans = connection.begin()
            try:
                # Drop the table
                connection.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                trans.commit()
                return True, f"Table '{table_name}' deleted successfully"
            except Exception as e:
                trans.rollback()
                return False, f"Error deleting table: {str(e)}"
    except Exception as e:
        return False, f"Error connecting to database: {str(e)}" 