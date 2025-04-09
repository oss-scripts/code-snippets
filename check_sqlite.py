def check_sqlite_installation():
    """
    Check if SQLite is installed and working properly.
    Returns a tuple of (is_installed, version, error_message).
    """
    import sys
    import subprocess
    
    print("Checking SQLite installation...")
    
    # First, check if sqlite3 module is available in Python
    try:
        import sqlite3
        python_sqlite_version = sqlite3.sqlite_version
        print(f"✓ Python sqlite3 module found (version {python_sqlite_version})")
    except ImportError:
        return False, None, "Python sqlite3 module not found"
    
    # Try to create a test database and perform basic operations
    try:
        # Create a test connection and database
        print("Testing SQLite functionality...")
        conn = sqlite3.connect(":memory:")  # In-memory database for testing
        cursor = conn.cursor()
        
        # Create a simple table
        cursor.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
        
        # Insert data
        cursor.execute("INSERT INTO test_table (name) VALUES (?)", ("test_value",))
        
        # Query data
        cursor.execute("SELECT * FROM test_table")
        result = cursor.fetchall()
        
        if result and result[0][1] == "test_value":
            print(f"✓ SQLite database operations successful")
        else:
            return False, python_sqlite_version, "SQLite query returned unexpected results"
        
        # Check if we can perform a transaction
        conn.execute("BEGIN TRANSACTION")
        conn.execute("INSERT INTO test_table (name) VALUES (?)", ("transaction_test",))
        conn.commit()
        
        cursor.execute("SELECT COUNT(*) FROM test_table")
        count = cursor.fetchone()[0]
        
        if count == 2:
            print(f"✓ SQLite transactions working properly")
        else:
            return False, python_sqlite_version, "SQLite transaction test failed"
        
        conn.close()
        
        # Try to determine if SQLite CLI is installed
        try:
            result = subprocess.run(
                ["sqlite3", "--version"], 
                capture_output=True, 
                text=True,
                check=True
            )
            cli_version = result.stdout.strip()
            print(f"✓ SQLite command-line tool found (version {cli_version})")
        except (subprocess.SubprocessError, FileNotFoundError):
            print("ℹ️ SQLite command-line tool not found or not in PATH (optional)")
            # This is not a failure - the CLI tool is optional
        
        # Check filesystem access for database creation
        try:
            test_db_path = "sqlite_test_db.db"
            test_conn = sqlite3.connect(test_db_path)
            test_conn.close()
            import os
            os.remove(test_db_path)
            print("✓ File system access for database creation verified")
        except Exception as e:
            print(f"⚠️ File system access check: {str(e)}")
            print("  Note: This might be fine for in-memory databases")
        
        return True, python_sqlite_version, None
        
    except sqlite3.Error as e:
        return False, python_sqlite_version, f"SQLite error: {str(e)}"
    except Exception as e:
        return False, python_sqlite_version, f"Unexpected error: {str(e)}"

# Execute the check
if __name__ == "__main__":
    is_working, version, error = check_sqlite_installation()
    
    print("\n" + "="*50)
    if is_working:
        print(f"✅ SQLite is properly installed and functioning (version {version})")
        print("   You can proceed with implementing the analytics system.")
    else:
        print(f"❌ SQLite issue detected: {error}")
        print("   Please resolve this before implementing the analytics system.")
        
        # Provide some troubleshooting advice
        print("\nTroubleshooting tips:")
        print("1. Make sure sqlite3 is installed on your system")
        print("   - On Ubuntu/Debian: sudo apt-get install sqlite3")
        print("   - On CentOS/RHEL: sudo yum install sqlite")
        print("   - On macOS: Should be pre-installed, or use brew install sqlite")
        print("2. Check if Python's sqlite3 module is properly installed")
        print("3. Verify you have proper permissions for file creation in your working directory")
    print("="*50)
