from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.database import get_user_tables, delete_table

router = APIRouter(
    prefix="/api/tables",
    tags=["tables"],
)

@router.get("/", response_model=List[Dict[str, Any]])
def list_tables():
    """Get all user-created tables in the database"""
    return get_user_tables()

@router.delete("/{table_name}")
def remove_table(table_name: str):
    """Delete a table from the database"""
    success, message = delete_table(table_name)
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    return {"message": message} 