import sqlite3
import os

def test_sqlite():
    """Tests basic SQLite functionality."""

    db_file = "test_sqlite.db"

    try:
        # 1. Connect to the database (creates the file if it doesn't exist)
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        print(f"Successfully connected to SQLite database: {db_file}")

        # 2. Create a simple table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER
            )
        ''')
        conn.commit()
        print("Created 'users' table.")

        # 3. Insert some data
        users_data = [
            ("Alice", 30),
            ("Bob", 25),
            ("Charlie", 35)
        ]
        cursor.executemany("INSERT INTO users (name, age) VALUES (?, ?)", users_data)
        conn.commit()
        print("Inserted sample data into 'users' table.")

        # 4. Query the data
        cursor.execute("SELECT * FROM users")
        rows = cursor.fetchall()
        print("\nData from 'users' table:")
        for row in rows:
            print(row)

        # 5. Clean up (optional, but good practice)
        cursor.execute("DROP TABLE IF EXISTS users")
        conn.commit()
        print("Dropped 'users' table.")

    except sqlite3.Error as e:
        print(f"SQLite error occurred: {e}")
        return False
    finally:
        if conn:
            conn.close()
            print("Closed the database connection.")
        # Remove the test file
        if os.path.exists(db_file):
            os.remove(db_file)
            print(f"Removed the test database file: {db_file}")

    return True

if __name__ == "__main__":
    print("Starting SQLite test...")
    if test_sqlite():
        print("\nSQLite test completed successfully on your RHEL system.")
    else:
        print("\nSQLite test failed. Please check the error messages.")
