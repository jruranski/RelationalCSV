from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from app.database import create_tables
from app.routers import files, graph, tables

# Create tables
create_tables()

# Initialize FastAPI app
app = FastAPI(
    title="TPDIA DataLoom",
    description="A web app for uploading CSV files and generating SQL schemas",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Include routers
app.include_router(files.router)
app.include_router(graph.router)
app.include_router(tables.router)

# Root route - serve index.html
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Graph route - serve graph.html
@app.get("/graph")
async def graph_page(request: Request):
    return templates.TemplateResponse("graph.html", {"request": request})

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy"} 