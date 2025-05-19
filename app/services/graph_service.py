import os
import pandas as pd
import json
from typing import List, Dict, Any
import asyncio
from app.services.graph_definitions import app as graph_app

async def process_csvs_to_sql(file_paths: List[str], progress_callback=None):
    """
    Process CSV files into SQL using the graph defined in graph_definitions.py
    
    Args:
        file_paths: List of paths to CSV files
        progress_callback: Optional callback function to report progress
        
    Returns:
        dict: Dictionary with keys 'schemas', 'relationships', and 'sql_query'
    """
    # Helper function to safely call the progress callback
    async def report_progress(message):
        if progress_callback and callable(progress_callback):
            await progress_callback(message)
            # Allow WebSocket messages to be sent by yielding to the event loop
            await asyncio.sleep(0.01)  # Small delay to ensure messages are processed
    
    # Load CSV files into strings
    csv_strings = []
    file_names = []
    
    await report_progress(f"📊 Starting analysis of {len(file_paths)} CSV files")
    
    for csv_file_path in file_paths:
        try:
            df = pd.read_csv(csv_file_path)
            filename = os.path.basename(csv_file_path)
            file_names.append(filename)
            headers = df.columns.tolist()
            data = df.head(10).to_string(index=False)
            # Format CSV string with filename and data
            csv_string = f"{filename}\n  {','.join(headers)}\n{data}"
            csv_strings.append(csv_string)
            
            await report_progress(f"📄 Loaded CSV file: {filename} ({len(df)} rows, {len(headers)} columns)")
            column_types = {}
            for col in headers:
                try:
                    inferred_type = str(df[col].dtype)
                    column_types[col] = inferred_type
                except:
                    column_types[col] = "unknown"
            
            column_info = ", ".join([f"{col}: {typ}" for col, typ in column_types.items()])
            await report_progress(f"📋 Columns in {filename}: {column_info}")
        except Exception as e:
            await report_progress(f"❌ Error loading CSV file {csv_file_path}: {str(e)}")
            raise ValueError(f"Error loading CSV file {csv_file_path}: {str(e)}")
    
    # Prepare input for the graph
    graph_input = {
        "csv_strings": csv_strings,
        "all_csv_schemas": [],
        "all_relationships": []
    }
    
    # Track the processing steps and results
    results = {
        "schemas": [],
        "relationships": [],
        "sql_query": "",
        "steps": []
    }
    
    last_node = None
    current_schema = None
    
    # Stream the graph execution and collect results
    try:
        await report_progress("🚀 Starting AI-powered schema analysis")
            
        for output in graph_app.stream(graph_input, {"recursion_limit": 100}):
            for node_name, node_value in output.items():
                # Track the step
                step = f"Completed: {node_name}"
                results["steps"].append(step)
                
                # More detailed progress updates based on the node type
                if node_name == "start_new_csv" and "selected_csv" in node_value:
                    selected_csv = node_value.get("selected_csv", "")
                    # Extract the filename from the CSV string
                    filename = selected_csv.split("\n")[0] if "\n" in selected_csv else "unknown file"
                    await report_progress(f"🔍 Analyzing table structure for: {filename}")
                
                elif node_name == "convert_to_schema" and "new_csv_schema" in node_value:
                    try:
                        schema_json = node_value.get("new_csv_schema", "{}")
                        schema = json.loads(schema_json)
                        current_schema = schema
                        table_name = schema.get("table_name", "unknown table")
                        columns = schema.get("columns", [])
                        
                        # Format column details for display
                        column_details = []
                        for col in columns:
                            constraints = ", ".join(col.get("constraints", []))
                            column_details.append(f"{col['name']} ({col['type']}{' | ' + constraints if constraints else ''})")
                        
                        column_info = "; ".join(column_details)
                        await report_progress(f"📝 Created schema for table: {table_name}")
                        await report_progress(f"🏗️ Columns defined: {column_info}")
                    except Exception as e:
                        await report_progress(f"⚠️ Error parsing schema: {str(e)}")
                
                elif node_name == "find_relationships" and "all_relationships" in node_value:
                    if current_schema:
                        table_name = json.loads(current_schema).get("table_name", "unknown table") if isinstance(current_schema, str) else current_schema.get("table_name", "unknown table")
                        await report_progress(f"🔗 Finding relationships for table: {table_name}")
                    
                    relationships = node_value.get("all_relationships", [])
                    # Check if any new relationships were added
                    if last_node and last_node == "find_relationships" and len(relationships) > results.get("relationship_count", 0):
                        new_relationship_count = len(relationships) - results.get("relationship_count", 0)
                        results["relationship_count"] = len(relationships)
                        
                        # Get the newest relationships
                        new_relationships = relationships[-new_relationship_count:]
                        for rel in new_relationships:
                            if hasattr(rel, "relationship_name") and hasattr(rel, "table_name") and hasattr(rel, "relationship_type"):
                                rel_type = rel.relationship_type
                                rel_name = rel.relationship_name
                                from_table = rel.table_name
                                to_table = getattr(rel, "target_table", "unknown")
                                
                                if rel_type == "many-to-many":
                                    join_table = getattr(rel, "join_table", "unknown")
                                    await report_progress(f"🔄 Found {rel_type} relationship: {from_table} ↔ {join_table} ↔ {to_table} ({rel_name})")
                                else:
                                    await report_progress(f"➡️ Found {rel_type} relationship: {from_table} → {to_table} ({rel_name})")
                
                elif node_name == "generate_sql" and "sql_query" in node_value:
                    sql_query = node_value.get("sql_query", "")
                    table_count = sql_query.count("CREATE TABLE")
                    
                    await report_progress(f"📜 Generated SQL schema with {table_count} tables")
                    await report_progress("✅ Schema generation complete!")
                
                last_node = node_name
                
                await report_progress(f"✓ Completed step: {node_name}")
        
        # Extract the final results
        if "all_csv_schemas" in node_value:
            results["schemas"] = node_value["all_csv_schemas"]
            await report_progress(f"📊 Defined schemas for {len(results['schemas'])} tables")
        
        if "all_relationships" in node_value:
            # Convert relationship objects to dict for JSON serialization
            relationships = []
            for rel in node_value["all_relationships"]:
                if hasattr(rel, "model_dump"):
                    relationships.append(rel.model_dump())
                else:
                    relationships.append(rel)
            results["relationships"] = relationships
            await report_progress(f"🔗 Identified {len(results['relationships'])} relationships between tables")
        
        if "sql_query" in node_value:
            sql_query = node_value["sql_query"]
            # Clean up SQL query if needed (remove markdown formatting)
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            # Ensure each statement ends with a semicolon and has proper spacing
            sql_statements = []
            for stmt in sql_query.split(';'):
                stmt = stmt.strip()
                if stmt:
                    # Remove any trailing commas before the closing parenthesis
                    stmt = stmt.replace(",\n)", "\n)")
                    stmt = stmt.replace(", )", ")")
                    
                    # Add semicolon if missing
                    if not stmt.endswith(';'):
                        stmt += ';'
                    sql_statements.append(stmt)
            
            # Join with proper spacing between statements
            sql_query = '\n\n'.join(sql_statements)
            
            results["sql_query"] = sql_query
            await report_progress(f"🎉 SQL query generation successful!")
            
    except Exception as e:
        await report_progress(f"❌ Error during graph execution: {str(e)}")
        raise ValueError(f"Error during graph execution: {str(e)}")
    
    return results 