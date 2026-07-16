import mysql.connector

current_connection = None


# -------------------------------------------------------
# Connect
# -------------------------------------------------------

def connect(host, port, database, username, password):

    return mysql.connector.connect(
        host=host,
        port=port,
        user=username,
        password=password,
        database=database
    )


# -------------------------------------------------------
# Test Connection
# -------------------------------------------------------

def test_connection(host, port, database, username, password):

    global current_connection

    try:

        current_connection = connect(
            host,
            port,
            database,
            username,
            password
        )

        return {
            "success": True,
            "message": "Connection Successful!"
        }

    except Exception as e:

        current_connection = None

        return {
            "success": False,
            "message": str(e)
        }


# -------------------------------------------------------
# Tables
# -------------------------------------------------------

def get_tables():

    global current_connection

    cursor = current_connection.cursor()

    cursor.execute("SHOW TABLES")

    tables = [row[0] for row in cursor.fetchall()]

    cursor.close()

    return tables


# -------------------------------------------------------
# Table Data
# -------------------------------------------------------

def get_table_data(table):

    global current_connection

    cursor = current_connection.cursor(dictionary=True)

    cursor.execute(f"SELECT * FROM `{table}` LIMIT 100")

    rows = cursor.fetchall()

    cursor.close()

    return rows


# -------------------------------------------------------
# Execute Any SQL
# -------------------------------------------------------

def execute_sql(sql):

    global current_connection

    cursor = current_connection.cursor(dictionary=True)

    cursor.execute(sql)

    rows = cursor.fetchall()

    cursor.close()

    return rows


# -------------------------------------------------------
# Database Schema
# -------------------------------------------------------

def get_database_schema():

    global current_connection

    cursor = current_connection.cursor(dictionary=True)

    cursor.execute("SHOW TABLES")

    tables = [row[f"Tables_in_{current_connection.database}"] for row in cursor.fetchall()]

    schema = {}

    for table in tables:

        cursor.execute(f"DESCRIBE `{table}`")

        schema[table] = cursor.fetchall()

    cursor.close()

    return schema