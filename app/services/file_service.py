import os
import shutil
import uuid
import pandas as pd
from sqlalchemy.orm import Session
from fastapi import UploadFile

from app.database import UploadedFile
from app.models.file import FileResponse, FilePreview

UPLOAD_DIR = "app/static/uploads"

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_file(file: UploadFile, db: Session) -> FileResponse:
    """Save an uploaded file to disk and database"""
    # Generate unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Read file information
    file_size = os.path.getsize(file_path)
    
    # Read CSV metadata
    try:
        df = pd.read_csv(file_path)
        row_count = len(df)
        columns = ",".join(df.columns.tolist())
    except Exception:
        row_count = 0
        columns = ""
    
    # Save to database
    db_file = UploadedFile(
        filename=unique_filename,
        original_filename=file.filename,
        file_path=file_path,
        file_size=file_size,
        row_count=row_count,
        columns=columns
    )
    
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    
    return FileResponse.from_orm(db_file)

def get_files(db: Session):
    """Get all files from database"""
    return db.query(UploadedFile).all()

def get_file(file_id: int, db: Session):
    """Get file by ID"""
    return db.query(UploadedFile).filter(UploadedFile.id == file_id).first()

def get_file_preview(file_id: int, db: Session) -> FilePreview:
    """Get file preview data"""
    file = get_file(file_id, db)
    if not file:
        return None
    
    try:
        df = pd.read_csv(file.file_path)
        headers = df.columns.tolist()
        # Get first 10 rows as sample data
        sample_data = df.head(10).values.tolist()
        
        return FilePreview(
            id=file.id,
            filename=file.original_filename,
            headers=headers,
            sample_data=sample_data,
            row_count=file.row_count
        )
    except Exception as e:
        print(f"Error previewing file: {e}")
        return None

def delete_file(file_id: int, db: Session) -> bool:
    """Delete a file from disk and database"""
    file = get_file(file_id, db)
    if not file:
        return False
    
    # Delete file from disk
    try:
        if os.path.exists(file.file_path):
            os.remove(file.file_path)
    except Exception as e:
        print(f"Error deleting file from disk: {e}")
    
    # Delete from database
    try:
        db.delete(file)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"Error deleting file from database: {e}")
        return False

def delete_all_files(db: Session) -> int:
    """Delete all files from disk and database"""
    files = get_files(db)
    count = 0
    
    for file in files:
        if delete_file(file.id, db):
            count += 1
    
    return count 