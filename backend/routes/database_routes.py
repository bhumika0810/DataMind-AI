from fastapi import APIRouter
from pydantic import BaseModel
from services.database_services import get_database_schema

from services.database_services import (
    test_connection,
    get_tables,
    get_table_data,
)

from services.database_ai_service import ask_database

router = APIRouter()


# ---------------------------------------------------
# Models
# ---------------------------------------------------

class DatabaseConnection(BaseModel):
    host: str
    port: int
    database: str
    username: str
    password: str


class DatabaseQuestion(BaseModel):
    question: str


# ---------------------------------------------------
# Test Connection
# ---------------------------------------------------

@router.post("/test")
def test_database(data: DatabaseConnection):

    return test_connection(
        data.host,
        data.port,
        data.database,
        data.username,
        data.password
    )


# ---------------------------------------------------
# Get Tables
# ---------------------------------------------------

@router.post("/tables")
def tables(data: DatabaseConnection):

    result = test_connection(
        data.host,
        data.port,
        data.database,
        data.username,
        data.password
    )

    if not result["success"]:
        return result

    return {
        "success": True,
        "tables": get_tables()
    }
# ---------------------------------------------------
# Read Table
# ---------------------------------------------------

@router.post("/table/{table_name}")
def read_table(table_name: str):

    return {
        "rows": get_table_data(table_name)
    }


# ---------------------------------------------------
# AI Chat
# ---------------------------------------------------

@router.post("/chat")
async def database_chat(data: DatabaseQuestion):

    schema = get_database_schema()
    

    return await ask_database(
        data.question,
        schema
    )