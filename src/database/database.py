## src/database/database.py:
import sqlite3
import pyodbc
import datetime
import logging
import time
import os
from config.config import settings

logger = logging.getLogger(__name__)

class DatabaseLogger:
    """
    Handles logging face detection events exclusively to the local SQLite database.
    This class is decoupled from the synchronization process.
    """
    def __init__(self):
        """Initialize connection to the local SQLite database."""
        self.local_db_path = "/app/outputs/local.db"
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.local_db_path), exist_ok=True)
        
        # Connect to SQLite with a timeout to prevent lock issues.
        self.local_conn = sqlite3.connect(self.local_db_path, check_same_thread=False, timeout=10)
        self.local_cursor = self.local_conn.cursor()
        self._create_local_tables()

    def _create_local_tables(self):
        """Create the 'faces' table in SQLite if it doesn't exist."""
        self.local_cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT,
                personnel_ids TEXT,
                frame_datetime TEXT,
                frame_data BLOB,
                synced INTEGER DEFAULT 0,
                memo TEXT DEFAULT '1'
            )
        """)
        self.local_conn.commit()

    def log_faces(self, camera_id, personnel_ids, frame_datetime, frame_data, memo="0"):
        """Log face detection results to the local SQLite database."""
        personnel_ids_str = ",".join(personnel_ids) if personnel_ids else ""
        
        try:
            self.local_cursor.execute(
                "INSERT INTO faces (camera_id, personnel_ids, frame_datetime, frame_data, synced, memo) VALUES (?, ?, ?, ?, 0, ?)",
                (str(camera_id), personnel_ids_str, frame_datetime.isoformat(), frame_data, memo)
            )
            self.local_conn.commit()
            logger.info(f"📝 Faces from camera '{camera_id}' logged locally to SQLite with memo '{memo}'")
        except Exception as e:
            logger.error(f"❌ Failed to write to local SQLite database: {e}")

    def close(self):
        """Close the SQLite database connection."""
        if self.local_conn:
            self.local_conn.close()
            logger.info("🔚 Local database connection closed.")


class DatabaseSynchronizer:
    """
    Handles synchronizing records from the local SQLite DB to the main SQL Server.
    This class is intended to be used by a separate sync service.
    """
    def __init__(self):
        """Initialize connections to both SQLite and SQL Server."""
        self.main_conn_str = (
            f'DRIVER={settings.SQL_DRIVER};'
            f'SERVER={settings.SQL_SERVER};'
            f'DATABASE={settings.SQL_DATABASE};'
            f'UID={settings.SQL_UID};'
            f'PWD={settings.SQL_PWD}'
        )
        self.local_db_path = "/app/outputs/local.db"
        self.local_conn = sqlite3.connect(self.local_db_path, timeout=10)
        self.local_cursor = self.local_conn.cursor()
        
        self.main_conn = None
        self.main_cursor = None
        
    def _connect_to_main_db(self):
        """Connect to the main SQL Server database."""
        try:
            if self.main_conn: # Close existing connection if any
                self.main_conn.close()
            self.main_conn = pyodbc.connect(self.main_conn_str, timeout=5)
            self.main_cursor = self.main_conn.cursor()
            logger.info("✅ Connected to SQL Server for synchronization.")
            return True
        except Exception as e:
            logger.warning(f"⚠️ SQL Server connection failed: {e}. Will retry on the next cycle.")
            self.main_conn = None
            self.main_cursor = None
            return False

    def synchronize(self):
        """Synchronize a batch of unsynced records from SQLite to SQL Server."""
        if not self.main_conn or not self.main_cursor:
            if not self._connect_to_main_db():
                logger.warning("⚠️ Main DB unavailable, skipping this sync cycle.")
                return

        batch_size = 50
        try:
            self.local_cursor.execute(f"SELECT * FROM faces WHERE synced = 0 ORDER BY id ASC LIMIT {batch_size}")
            records_to_sync = self.local_cursor.fetchall()
            
            if not records_to_sync:
                logger.info("✅ No unsynced records found. Database is up-to-date.")
                return

            logger.info(f"📋 Found {len(records_to_sync)} unsynced records to synchronize.")
            
            for record in records_to_sync:
                rec_id, cam_id, p_ids, dt_iso, f_data, _, memo_val = record
                self.main_cursor.execute(
                    "EXEC aiStpInsertFrameFace @cameraId=?, @employeeCodes=?, @frameDateTime=?, @frame=?, @memo=?",
                    (cam_id, p_ids, datetime.datetime.fromisoformat(dt_iso), f_data, memo_val)
                )
            
            self.main_conn.commit()
            
            synced_ids = [rec[0] for rec in records_to_sync]
            self.local_cursor.execute(f"UPDATE faces SET synced = 1 WHERE id IN ({','.join('?'*len(synced_ids))})", synced_ids)
            self.local_conn.commit()
            logger.info(f"✅ Synced batch of {len(records_to_sync)} records successfully.")

        except pyodbc.Error as e:
            logger.error(f"❌ Database error during sync batch: {e}. Will attempt to rollback and reconnect.")
            if self.main_conn:
                self.main_conn.rollback()
            # Force a reconnect on the next cycle
            self._connect_to_main_db() 
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred during sync: {e}")
            if self.main_conn:
                self.main_conn.rollback()

    def close(self):
        """Close all database connections."""
        if self.main_conn:
            self.main_conn.close()
        if self.local_conn:
            self.local_conn.close()
        logger.info("🔚 All synchronizer database connections closed.")