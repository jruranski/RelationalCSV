import pandas as pd
from typing import Dict, List, Tuple
import re
from sqlalchemy.orm import Session

from app.database import execute_sql
from app.services.file_service import get_file

def infer_column_type(column_values: pd.Series) -> str:
    """Infer SQL column type from pandas series data"""
    # Check if all values are missing
    if column_values.isna().all():
        return "TEXT"
    
    # Remove NaN values for type checking
    non_na_values = column_values.dropna()
    
    # If no values left, default to TEXT
    if len(non_na_values) == 0:
        return "TEXT"
    
    # Try to convert to integer
    try:
        pd.to_numeric(non_na_values, downcast='integer')
        # Check if all values can be integers
        if all(float(x).is_integer() for x in non_na_values if pd.notna(x)):
            return "INTEGER"
    except:
        pass
    
    # Try to convert to float
    try:
        pd.to_numeric(non_na_values)
        return "REAL"
    except:
        pass
    
    # Default to TEXT
    return "TEXT"

def clean_column_name(name: str) -> str:
    """Clean column name to valid SQL identifier"""
    # Replace spaces and special chars with underscore
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure name starts with letter or underscore
    if not cleaned[0].isalpha() and cleaned[0] != '_':
        cleaned = 'col_' + cleaned
    return cleaned.lower()

def generate_create_table_sql(file_id: int, table_name: str, db: Session) -> Tuple[str, Dict]:
    """Generate SQL CREATE TABLE statement from CSV file"""
    file = get_file(file_id, db)
    if not file:
        return None, {}
    
    try:
        # Read CSV
        df = pd.read_csv(file.file_path)
        
        # Clean column names
        df.columns = [clean_column_name(col) for col in df.columns]
        
        # Infer column types
        column_types = {}
        for column in df.columns:
            column_types[column] = infer_column_type(df[column])
        
        # Build CREATE TABLE statement
        columns_sql = []
        for column, dtype in column_types.items():
            columns_sql.append(f"`{column}` {dtype}")
        
        create_table_sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` (\n"
        create_table_sql += ",\n".join(columns_sql)
        create_table_sql += "\n);"
        
        return create_table_sql, column_types
    except Exception as e:
        print(f"Error generating schema: {e}")
        return None, {}

def execute_create_table(sql: str) -> bool:
    """Execute CREATE TABLE SQL statement"""
    try:
        execute_sql(sql)
        return True
    except Exception as e:
        print(f"Error executing SQL: {e}")
        return False 