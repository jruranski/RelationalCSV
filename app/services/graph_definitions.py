from .create_llm import create_llm

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
import sys
from pathlib import Path

# from .DocumentFormatter import DocumentFormatter
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))
from typing_extensions import TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

from langgraph.checkpoint.memory import MemorySaver
import json
from pathlib import Path
import os
import concurrent.futures
import pandas as pd

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Prompt

# LLM
llm = create_llm(temperature=0.7) 


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain

# Run
# generation = rag_chain.invoke({"context": docs, "question": question}) # Example usage



class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
# LLM with function call
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

### Hallucination Grader


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# LLM with function call
llm = create_llm(temperature=0.2)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
# hallucination_grader.invoke({"documents": docs, "generation": generation}) # Example usage

# ==================
class SchemaColumn(BaseModel):
    """SQL Column"""
    name: str = Field(description="Column name")
    type: str = Field(description="Column type")
    constraints: List[str] = Field(description="List of constraints, e.g. 'NOT NULL', 'UNIQUE', 'PRIMARY KEY', 'FOREIGN KEY'")

class Schema(BaseModel):
    """SQL Schema"""
    table_name: str = Field(description="Name of the table")
    columns: List[SchemaColumn] = Field(description="List of columns")

class RelationshipSchema(BaseModel):
    """Relationship between two tables"""
    table_name: str = Field(description="Name of the table")
    foreign_key: str = Field(description="Foreign key of the table")
    relationship_type: Literal["null", "one-to-one", "many-to-one", "many-to-many"] = Field(description="Relationship type: null, one-to-one, many-to-one, or many-to-many")
    relationship_name: str = Field(description="Relationship name")
    relationship_description: str = Field(description="Relationship description")
    # Fields needed for many-to-many relationships
    join_table: Optional[str] = Field(default=None, description="Join table name for many-to-many relationships")
    target_table: Optional[str] = Field(default=None, description="Target table in the relationship")
    target_key: Optional[str] = Field(default=None, description="Key in the target table")
class NewRelationships(BaseModel):
    """New relationships between tables"""
    relationships: Optional[List[RelationshipSchema]] = Field(default=None, description="List of relationships")
schema_llm = llm.with_structured_output(Schema)

schema_system_prompt = """
You are a master database engineer. Your job is to create SQLite table schemas based on snippets of CSV Files.
"""
schema_generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", schema_system_prompt),
        ("human", "CSV File: \n\n {csv_file} \n\n Create a schema for the table in the specified format."),
    ]
)

schema_generator = schema_generation_prompt | schema_llm
# ==================

relationship_llm = llm.with_structured_output(NewRelationships)
relationship_system_prompt = """
You are a master database engineer. Your job is to find NEW relationships between tables based on snippets of CSV Files.

For each relationship type:
1. For "null" relationships: Set the relationship type accordingly when no relationship exists.
2. For "one-to-one" relationships: Identify when a record in one table corresponds to exactly one record in another table.
3. For "many-to-one" relationships: Identify when multiple records in one table relate to a single record in another.
4. For "many-to-many" relationships: Identify relationships requiring a join table, and specify the join_table, target_table, and target_key fields.

Be precise with foreign keys and clearly describe the relationship. You can output multiple relationships if needed.
If there are absolutely no relationships output NULL.
Tabels can have a relationship with itself but only on different columns.
"""
relationship_generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", relationship_system_prompt),
        ("human", "Please find NEW relationships between tables between the following schemas: \n\n {schemas} and this new schema: \n\n {new_schema} these are all of the previously found relationships: \n\n {relationships}. For many-to-many relationships, explicitly specify the join_table, target_table, and target_key fields. For the other types, focus on proper table_name and foreign_key identification."),
    ]
)

relationship_generator = relationship_generation_prompt | relationship_llm




# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

llm = create_llm( temperature=0.15)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
# answer_grader.invoke({"question": question, "generation": generation}) # Example usage


class GradeRelationships(BaseModel):
    """Binary score to assess relationships between tables."""
    binary_score: str = Field(
        description="Relationship is correct, 'yes' or 'no'"
    )

relationship_llm = llm.with_structured_output(GradeRelationships)

relationship_system_prompt = """
You are a master database engineer. Your job is to grade the relationships between tables.

Give a binary score 'yes' or 'no'. Yes' means that the relationship is correct.
"""

relationship_generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", relationship_system_prompt),
        ("human", "Relationship: \n\n {relationship} \n\n Is it correct? Give a binary score 'yes' or 'no'."),
    ]
)
relationship_grader = relationship_generation_prompt | relationship_llm


sql_generator_system_prompt = """
You are a master SQLite database engineer. Your job is to generate a SQL to create a new database with the tables and relationships that will be provided to you.
Given the input data and refinement hints, create syntatically correct SQLite query to run. Output only the SQL query, no other text. Always complete the query, include all tabels and relationships.
The database should be created in the following format:
Remember to use IF NOT EXISTS when creating tables.
ALWAYS VERIFY THAT THE TABLES ARE IN THE CORRECT ORDER AND THAT THE RELATIONSHIPS ARE CORRECT.


"""
sql_generator_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sql_generator_system_prompt),
        ("human", "Create a complete query to create a database with the following tables and relationships: Tables: \n\n {tables} \n\n Relationships: \n\n {relationships} \n\n Generate a SQL query. Output only the SQL query, no other text. Keep in mind: {hint}"),
    ]
)
sql_llm = llm | StrOutputParser()
sql_generator = sql_generator_prompt | sql_llm

class GradeSQL(BaseModel):
    """Binary score to assess SQL query and a hint for the generation model"""
    binary_score: str = Field(
        description="SQL query is correct, 'yes' or 'no'"
    )
    hint: str = Field(
        description="Hint for the generation model to improve the query"
    )

sql_reviewer_system_prompt = """
You are a master SQLite database engineer. Your job is to review a SQL query to create a database with the tables and relationships that will be provided to you.

Give a binary score 'yes' or 'no'. Yes' means that the query is correct. If it is not correct provide a hint to the model that created the query.
"""
sql_reviewer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sql_reviewer_system_prompt),
        ("human", "Review the following SQL query: \n\n {sql_query} \n\n Is it correct? Give a binary score 'yes' or 'no'. If it is not correct provide a hint to the model that created the query."),
    ]
)
review_sql_llm = llm.with_structured_output(GradeSQL)
sql_reviewer = sql_reviewer_prompt | review_sql_llm
# question_rewriter.invoke({"question": question}) # Example usage

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        csv_strings: list of csv strings/files
        generation: LLM generation
        documents: list of documents
    """

    csv_strings: List[str]
    new_csv_schema: str
    all_csv_schemas: List[str]
    new_relationships: List[RelationshipSchema]
    all_relationships: List[RelationshipSchema]
    filtered_relationships: List[RelationshipSchema]
    sql_query: str
    selected_csv: str
    hint: str

# def grade_relationships(state):
#     """
#     Grade the relationships between tables.
#     """
#     print("---GRADE RELATIONSHIPS---")
#     relationships = state["new_relationships"]
#     def grade_relationship(relationship):
#         score = relationship_grader.invoke({"relationship": relationship})
#         grade = score["binary_score"]
#         if grade == "yes":
#             print("---GRADE: RELATIONSHIP IS CORRECT---")
#             return relationship
#         else:
#             print("---GRADE: RELATIONSHIP IS INCORRECT---")
#             return None
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         results = list(executor.map(grade_relationship, relationships))
#     filtered_relationships = [r for r in results if r is not None]
#     state["filtered_relationships"] = filtered_relationships
#     return state

# def decide_to_regenerate_relationships(state: GraphState):
#     """
#     Decide whether to regenerate the relationships between tables.
#     """
#     print("---DECIDE TO REGENERATE RELATIONSHIPS---")
#     filtered_relationships = state["filtered_relationships"]
#     if len(filtered_relationships) == 0 and len(state["new_relationships"]) > 0:
#         print("---NO FILTERED RELATIONSHIPS, REGENERATING---")
#         return "find_relationships"
#     else:
#         print("---FILTERED RELATIONSHIPS, CONTINUING---")
#         return "continue"

#     Determines whether the retrieved documents are relevant to the question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Updates documents key with only filtered relevant documents
#     """
# def grade_documents(state):
#     """
#     Determines whether the retrieved documents are relevant to the question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Updates documents key with only filtered relevant documents
#     """

#     print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
#     question = state["question"]
#     documents = state["documents"]

#     def grade_document(d):
#         score = retrieval_grader.invoke(
#             {"question": question, "document": d.page_content}
#         )
#         grade = score.binary_score
#         if grade == "yes":
#             print("---GRADE: DOCUMENT RELEVANT---")
#             return d
#         else:
#             print("---GRADE: DOCUMENT NOT RELEVANT---")
#             return None

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         results = list(executor.map(grade_document, documents))

#     filtered_docs = [d for d in results if d is not None]

#     return {"documents": filtered_docs, "question": question}


# def transform_query(state):
#     """
#     Transform the query to produce a better question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Updates question key with a re-phrased question
#     """

#     print("---TRANSFORM QUERY---")
#     question = state["question"]
#     documents = state["documents"]
#     context =""# _context_for_mode(RetrievalMode.STPP)
#     system = f"""You a question re-writer that converts an input question to a better version that is optimized \n 
#      for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. ALWAYS OUTPUT QUESTION IN POLISH LANGUAGE!!!
#       You are a question rewriter at TFKable - Cable Production company. 
#       Here is context for the context of the documents we are working with: \n\n {context} \n\n
#      """
#     re_write_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system),
#             (
#                 "human",
#                 "Here is the initial question: \n\n {question} \n Formulate an improved question.",
#             ),
#         ]
#     )

#     question_rewriter = re_write_prompt | llm | StrOutputParser()
#     # Re-write question
#     better_question = question_rewriter.invoke({"question": question})
#     print(f"---BETTER QUESTION: {better_question}---")
#     return {"documents": documents, "question": better_question}


# ### Edges


# def decide_to_generate(state):
#     """
#     Determines whether to generate an answer, or re-generate a question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         str: Binary decision for next node to call
#     """

#     print("---ASSESS GRADED DOCUMENTS---")
#     state["question"]
#     filtered_documents = state["documents"]

#     if not filtered_documents:
#         # All documents have been filtered check_relevance
#         # We will re-generate a new query
#         print(
#             "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
#         )
#         return "transform_query"
#     else:
#         # We have relevant documents, so generate answer
#         print("---DECISION: GENERATE---")
#         return "generate"


# def grade_generation_v_documents_and_question(state):
#     """
#     Determines whether the generation is grounded in the document and answers question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         str: Decision for next node to call
#     """

#     print("---CHECK HALLUCINATIONS---")
#     question = state["question"]
#     documents = state["documents"]
#     generation = state["generation"]

#     score = hallucination_grader.invoke(
#         {"documents": documents, "generation": generation}
#     )
#     grade = score.binary_score

#     # Check hallucination
#     if grade == "yes":
#         print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
#         # Check question-answering
#         print("---GRADE GENERATION vs QUESTION---")
#         score = answer_grader.invoke({"question": question, "generation": generation})
#         grade = score.binary_score
#         if grade == "yes":
#             print("---DECISION: GENERATION ADDRESSES QUESTION---")
#             return "useful"
#         else:
#             print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
#             return "not useful"
#     else:
#         print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
#         return "not supported"

def start_new_csv(state: GraphState):
    """
     Start the execution of the graph.
    """
    print("---START EXECUTION---")
    csv_strings = state["csv_strings"]

    # pick one csv string to start and remove it from the csv_strings
    state["selected_csv"] = csv_strings[0]
    state["csv_strings"] = csv_strings[1:]
    print(f"---STARTING NEW CSV: {state['selected_csv']}---")

    return state


def convert_to_schema(state: GraphState):
    selected_csv = state["selected_csv"]
    schema = schema_generator.invoke({"csv_file": selected_csv})
    print(f"OUTPUT NEW SCHEMA {schema} of type {type(schema)}")
    if isinstance(schema, Schema):
        state["new_csv_schema"] = schema.model_dump_json()
        state["all_csv_schemas"].append(schema.model_dump_json())
        print(f"---NEW SCHEMA: {state['new_csv_schema']}---")
    else:
        raise ValueError("Schema generator did not return a dictionary.")
    return state

def find_relationships(state: GraphState):
    """
    Find relationships between tables.
    """
    print("---FIND RELATIONSHIPS---")
    all_csv_schemas = state["all_csv_schemas"]
    new_csv_schema = state["new_csv_schema"]
    relationships = state["all_relationships"]
    relationships_json = [relationship.model_dump_json() for relationship in relationships]
    
    if len(all_csv_schemas) == 1:
        print("only one schema, no relationships to find")
        return state
        
    new_relationships = relationship_generator.invoke({"schemas": all_csv_schemas, "new_schema": new_csv_schema, "relationships": relationships_json})

    if isinstance(new_relationships, NewRelationships):
        print(f"---NEW RELATIONSHIPS: {new_relationships}---")
        new_relationships = new_relationships.relationships
        if new_relationships is None:
            print("---NO NEW RELATIONSHIPS FOUND, SKIPPING---")
            return state
        for relationship in new_relationships:
            if relationship.relationship_type == "null":
                print("---NO NEW RELATIONSHIP FOUND, SKIPPING---")
                return state
        # Check if it's a many-to-many relationship that might need additional relationships
            if relationship.relationship_type == "many-to-many" and relationship.target_table:
                state["all_relationships"].append(relationship)
            
            # Potentially create the reverse relationship if appropriate
            if relationship.target_table and relationship.target_key:
                print("---CREATING REVERSE RELATIONSHIP FOR MANY-TO-MANY---")
                # The relationship goes both ways in many-to-many
                reverse_relationship = RelationshipSchema(
                    table_name=relationship.target_table,
                    foreign_key=relationship.target_key,
                    relationship_type="many-to-many",
                    relationship_name=f"{relationship.relationship_name}_reverse",
                    relationship_description=f"Reverse of: {relationship.relationship_description}",
                    join_table=relationship.join_table,
                    target_table=relationship.table_name,
                    target_key=relationship.foreign_key
                )
                state["all_relationships"].append(reverse_relationship)
        else:
            state["all_relationships"].append(relationship)
            
        print(f"---ALL RELATIONSHIPS: {state['all_relationships']}---")
    elif new_relationships is None:
        print("---NO NEW RELATIONSHIPS FOUND---")
    else:
        print(f"---NEW WRONG RELATIONSHIPS: {new_relationships}---")
        raise ValueError("Relationship generator did not return a dictionary.")
    return state

def decide_to_start_new_csv(state: GraphState):
    """
    Decide whether to start a new schema.
    """
    print("---DECIDE TO START NEW SCHEMA---")
    csv_files = state["csv_strings"]
    if len(csv_files) == 0:
        print("---NO CSV FILES LEFT, GENERATE SQL---")
        return "generate_sql"
    else:
        print("---MORE CSV FILES LEFT, START NEW CSV---")
        return "start_new_csv"

# Define the graph

def generate_sql(state: GraphState):
    """
    Generate a SQL query to create a database with the tables and relationships that will be provided to you.
    """
    print("---GENERATE SQL---")
    hint = state["hint"] if "hint" in state else ""
    print(f"Hint: {hint}")
    tables = "\n".join(state["all_csv_schemas"])
    relationships = "\n".join([relationship.model_dump_json() for relationship in state["all_relationships"]])
    sql_query = sql_generator.invoke({"tables": tables, "relationships": relationships, "hint": hint})
    state["sql_query"] = sql_query
    print(f"---SQL QUERY: {state['sql_query']}---")
    return state

def review_sql(state: GraphState):
    """
    Review the SQL query to create a database with the tables and relationships that will be provided to you.
    """
    print("---REVIEW SQL---")
    sql_query = state["sql_query"]
    grade: GradeSQL = sql_reviewer.invoke({"sql_query": sql_query}) # type: ignore 
    if grade.binary_score == "yes":
        print("---SQL QUERY IS CORRECT---")
        state["hint"] = ""
        return state
    else:
        print("---SQL QUERY IS INCORRECT---")
        print(f"---HINT: {grade.hint}---")
        state["hint"] = grade.hint
        return state 

def decide_to_end(state: GraphState):
    """
    Decide whether to end the workflow.
    """
    print("---DECIDE TO END---")
    if state["hint"] == "":
        print("---NO HINT, ENDING---")
        return "end"
    else:
        print("---HINT, CONTINUING---")
        return "generate_sql"

from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

workflow.add_node("start_new_csv", start_new_csv)
workflow.add_node("convert_to_schema", convert_to_schema)
workflow.add_node("find_relationships", find_relationships)
workflow.add_node("decide_to_start_new_csv", decide_to_start_new_csv)
workflow.add_node("generate_sql", generate_sql)
workflow.add_node("review_sql", review_sql)

workflow.add_edge(START, "start_new_csv")
workflow.add_edge("start_new_csv", "convert_to_schema")
workflow.add_edge("convert_to_schema", "find_relationships")
workflow.add_conditional_edges(
    "find_relationships",
    decide_to_start_new_csv,
    {
        "generate_sql": "generate_sql",
        "start_new_csv": "start_new_csv",
    },

)
workflow.add_edge("generate_sql", "review_sql")
workflow.add_conditional_edges(
    "review_sql",
    decide_to_end,
    {
        "generate_sql": "generate_sql",
        "end": END
    },
)

# Define the nodes
# workflow.add_node("retrieve", retrieve)  # retrieve
# workflow.add_node("grade_documents", grade_documents)  # grade documents
# workflow.add_node("generate", generate)  # generatae
# workflow.add_node("transform_query", transform_query)  # transform_query

# # Build graph
# workflow.add_edge(START, "retrieve")
# workflow.add_edge("retrieve", "grade_documents")
# workflow.add_conditional_edges(
#     "grade_documents",
#     decide_to_generate,
#     {
#         "transform_query": "transform_query",
#         "generate": "generate",
#     },
# )
# workflow.add_edge("transform_query", "retrieve")
# workflow.add_conditional_edges(
#     "generate",
#     grade_generation_v_documents_and_question,
#     {
#         "not supported": "generate",
#         "useful": END,
#         "not useful": "transform_query",
#     },
# )
# memory = MemorySaver()
# Compile
app = workflow.compile()


from pprint import pprint

# measure time
import time

# while True:

#     question = input("Please input your question: ")
#     if question == "exit":
#         break

#     start = time.time()
#     # Run
    # config = {
    #         "configurable": {"thread_id": str(123)},
    #     }
    # inputs = {"question": question}
    # for output in app.stream(inputs, config=config):
    #     for key, value in output.items():
    #         # Node
    #         pprint(f"Node '{key}':")
    #         # Optional: print full state at each node
    #         # pprint(value, indent=2, width=80, depth=None)
    #     pprint("\n---\n")

#     # Final generation
#     pprint(value["generation"])
#     end = time.time()
#     print(f"Time elapsed: {end - start} seconds")

def csv_to_sql(csv_file_paths):
     # load csv files
    csv_files = []
    csv_filenames = []
    for csv_file_path in csv_file_paths:
        df = pd.read_csv(csv_file_path)
        csv_filenames.append(csv_file_path.split("/")[-1])
        csv_files.append(df)


    # prepare csv headers + 10 rows of data in strings
    csv_strings = []
    for i, df in enumerate(csv_files):
        headers = df.columns.tolist()
        data = df.head(10).to_string(index=False)
        # add the filename to the string 
        csv_string = f"{csv_filenames[i]}\n  {','.join(headers)}\n{data}"
        print(csv_string)
        csv_strings.append(csv_string)

    return csv_strings

def test():
    csv_paths = [
        "/Users/jruranski/Developer/TPDIA_DataLoom/tests/test_bike_store_db/brands.csv",
        "/Users/jruranski/Developer/TPDIA_DataLoom/tests/test_bike_store_db/categories.csv",
        "/Users/jruranski/Developer/TPDIA_DataLoom/tests/test_bike_store_db/customers.csv",
        "/Users/jruranski/Developer/TPDIA_DataLoom/tests/test_bike_store_db/order_items.csv",
        "/Users/jruranski/Developer/TPDIA_DataLoom/tests/test_bike_store_db/orders.csv",
        "/Users/jruranski/Developer/TPDIA_DataLoom/tests/test_bike_store_db/products.csv",
        "/Users/jruranski/Developer/TPDIA_DataLoom/tests/test_bike_store_db/staffs.csv",
        "/Users/jruranski/Developer/TPDIA_DataLoom/tests/test_bike_store_db/stocks.csv",
        "/Users/jruranski/Developer/TPDIA_DataLoom/tests/test_bike_store_db/stores.csv"
    ]
    input = {"csv_strings": csv_to_sql(csv_paths), "all_csv_schemas": [], "all_relationships": []}
    
    # output =  app.invoke(input, config=config)
    for output in app.stream(input, {"recursion_limit": 100}):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint(value, indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    if "generation" in value:
        pprint(value["generation"])
    print("CSV Schemas:")
    print(value.get("all_csv_schemas", []))
    print("Relationships:")
    print(value.get("all_relationships", []))
    print(f"Number of relationships: {len(value.get('all_relationships', []))}")
    print(f"SQL Query: {value.get('sql_query', '')}")

# test()

if __name__ == "__main__":
    test()