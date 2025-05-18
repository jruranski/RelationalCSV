from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from app.database import get_db, get_db_schema
from app.models.file import FileResponse, FilePreview
from app.services import file_service, schema_service

router = APIRouter(
    prefix="/api/files",
    tags=["files"],
)

@router.post("/", response_model=FileResponse)
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a new CSV file"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    return file_service.save_file(file, db)

@router.get("/", response_model=List[FileResponse])
def get_files(db: Session = Depends(get_db)):
    """Get all uploaded files"""
    return file_service.get_files(db)

@router.get("/{file_id}", response_model=FileResponse)
def get_file(file_id: int, db: Session = Depends(get_db)):
    """Get file by ID"""
    file = file_service.get_file(file_id, db)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    return file

@router.get("/{file_id}/preview", response_model=FilePreview)
def get_file_preview(file_id: int, db: Session = Depends(get_db)):
    """Get file preview with sample data"""
    preview = file_service.get_file_preview(file_id, db)
    if not preview:
        raise HTTPException(status_code=404, detail="File not found or cannot be previewed")
    return preview

@router.post("/{file_id}/schema")
def generate_schema(
    file_id: int, 
    table_name: str = Form(...),
    db: Session = Depends(get_db)
):
    """Generate SQL schema from CSV file"""
    file = file_service.get_file(file_id, db)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    sql, column_types = schema_service.generate_create_table_sql(file_id, table_name, db)
    if not sql:
        raise HTTPException(status_code=400, detail="Failed to generate schema")
    
    return {
        "sql": sql,
        "column_types": column_types,
    }

@router.post("/{file_id}/execute-schema")
def execute_schema(
    file_id: int, 
    table_name: str = Form(...),
    db: Session = Depends(get_db)
):
    """Generate and execute SQL schema from CSV file"""
    file = file_service.get_file(file_id, db)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    sql, column_types = schema_service.generate_create_table_sql(file_id, table_name, db)
    if not sql:
        raise HTTPException(status_code=400, detail="Failed to generate schema")
    
    success = schema_service.execute_create_table(sql)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to execute SQL schema")
    
    return {
        "message": f"Table '{table_name}' created successfully",
        "sql": sql,
        "column_types": column_types,
    }

@router.delete("/{file_id}")
def delete_file(file_id: int, db: Session = Depends(get_db)):
    """Delete a file by ID"""
    success = file_service.delete_file(file_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="File not found or could not be deleted")
    
    return {"message": "File deleted successfully"}

@router.delete("/")
def delete_all_files(db: Session = Depends(get_db)):
    """Delete all files"""
    count = file_service.delete_all_files(db)
    
    return {"message": f"{count} files deleted successfully"}

@router.get("/schema/database", response_model=List[Dict[str, Any]])
def get_database_schema():
    """Get the current database schema"""
    return get_db_schema() 