from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
from sqlalchemy.orm import Session
import json
import asyncio

from app.database import get_db, execute_sql
from app.services import file_service, graph_service

router = APIRouter(
    prefix="/api/graph",
    tags=["graph"],
)

# Keep track of active websocket connections
active_connections = {}

@router.post("/process")
async def process_uploaded_files(file_ids: List[int], db: Session = Depends(get_db)):
    """
    Process uploaded files with the graph to generate SQL
    """
    # Validate files exist
    files = []
    for file_id in file_ids:
        file = file_service.get_file(file_id, db)
        if not file:
            raise HTTPException(status_code=404, detail=f"File with id {file_id} not found")
        files.append(file)
    
    # Get file paths
    file_paths = [file.file_path for file in files]
    
    try:
        # Process files synchronously (WebSocket version will handle progress)
        results = await graph_service.process_csvs_to_sql(file_paths)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time progress updates during graph processing
    """
    await websocket.accept()
    active_connections[client_id] = websocket
    
    try:
        while True:
            # Wait for commands from the client
            data = await websocket.receive_text()
            command = json.loads(data)
            
            if command["action"] == "process":
                file_ids = command["file_ids"]
                
                # Validate files
                files = []
                db = next(get_db())
                for file_id in file_ids:
                    file = file_service.get_file(file_id, db)
                    if not file:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"File with id {file_id} not found"
                        })
                        break
                    files.append(file)
                
                # Get file paths
                file_paths = [file.file_path for file in files]
                
                # Define progress callback
                async def progress_callback(message):
                    # Send update immediately
                    await websocket.send_json({
                        "type": "progress",
                        "message": message
                    })
                
                try:
                    # Send initial processing message
                    await websocket.send_json({
                        "type": "progress",
                        "message": "🚀 Starting processing..."
                    })
                    
                    # Process files with progress updates in background task
                    results = await graph_service.process_csvs_to_sql(file_paths, progress_callback)
                    
                    # Send final results
                    await websocket.send_json({
                        "type": "results",
                        "data": results
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
            
            elif command["action"] == "execute_sql":
                sql_query = command["sql_query"]
                
                # Debug logging for SQL commands
                print(f"Executing SQL query:\n{sql_query}")
                
                try:
                    # Execute the SQL query
                    results = execute_sql(sql_query)
                    num_statements = len(results)
                    
                    print(f"Successfully executed {num_statements} SQL statements")
                    
                    await websocket.send_json({
                        "type": "sql_executed",
                        "message": f"SQL query executed successfully. {num_statements} statement(s) processed."
                    })
                except Exception as e:
                    error_message = str(e)
                    # Simplify the error message for the UI
                    if "Error executing statement:" in error_message:
                        parts = error_message.split("\nError:", 1)
                        if len(parts) > 1:
                            error_message = f"SQL Error: {parts[1].strip()}"
                    
                    await websocket.send_json({
                        "type": "error",
                        "message": error_message
                    })
            
            elif command["action"] == "ping":
                # Simple ping to keep connection alive
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        if client_id in active_connections:
            del active_connections[client_id]
    except Exception as e:
        if client_id in active_connections:
            del active_connections[client_id]
        print(f"WebSocket error: {str(e)}")

@router.post("/execute-sql")
async def execute_sql_query(sql_query: str):
    """
    Execute a SQL query directly
    """
    try:
        results = execute_sql(sql_query)
        num_statements = len(results)
        return {"message": f"SQL query executed successfully. {num_statements} statement(s) processed."}
    except Exception as e:
        error_message = str(e)
        # Simplify the error message for the UI
        if "Error executing statement:" in error_message:
            parts = error_message.split("\nError:", 1)
            if len(parts) > 1:
                error_message = f"SQL Error: {parts[1].strip()}"
                
        raise HTTPException(status_code=500, detail=error_message) 