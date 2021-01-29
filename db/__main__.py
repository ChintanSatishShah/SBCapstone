"""Create a new instance of SQLite database, if it doesn't exist"""
import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
        print("SQLite DB created successfully")
    except Error as error:
        print("Error in creating SQLite DB:")
        print(error)
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    #DB_FILE_PATH = r'.\db\Test.db'
    DB_FILE_PATH = r'.\db\StockSeasonality.db'
    create_connection(DB_FILE_PATH)
    