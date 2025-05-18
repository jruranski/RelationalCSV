from pydantic import BaseModel
from typing import Optional, List
import datetime

class FileBase(BaseModel):
    filename: str
    
class FileCreate(FileBase):
    pass

class FileResponse(FileBase):
    id: int
    original_filename: str
    file_path: str
    upload_time: datetime.datetime
    file_size: int
    row_count: Optional[int] = None
    columns: Optional[str] = None
    
    model_config = {"from_attributes": True}

class FilePreview(BaseModel):
    id: int
    filename: str
    headers: List[str]
    sample_data: List[List[str]]
    row_count: int 