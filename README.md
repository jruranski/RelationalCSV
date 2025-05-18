# TPDIA DataLoom

A simple web application for uploading CSV files and generating SQL schemas.

## Features

- Upload CSV files
- View uploaded files
- Preview file details
- Generate SQL schema from CSV files

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   uvicorn app.main:app --reload
   ```

4. Open your browser and navigate to `http://localhost:8000`

## Project Structure

- `app/`: Main application package
  - `main.py`: FastAPI entry point
  - `database.py`: SQLite database connection
  - `models/`: Pydantic models
  - `routers/`: API routes
  - `services/`: Business logic
  - `static/`: Static files and uploaded files
  - `templates/`: HTML templates 