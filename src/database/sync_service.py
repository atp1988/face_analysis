## src/database/sync_service.py:
import time
import logging
import sys
from database import DatabaseSynchronizer

# Configure logging for the sync service
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SyncService")

def main():
    """
    Main function to run the synchronization service.
    This service runs in an infinite loop, periodically syncing data
    from the local SQLite database to the main SQL Server.
    """
    logger.info("🚀 Starting Synchronization Service...")
    
    # This object specifically handles the connection and synchronization logic.
    synchronizer = DatabaseSynchronizer()
    sync_interval_seconds = 30  # Run sync every 30 seconds

    logger.info(f"🔄 Service will attempt synchronization every {sync_interval_seconds} seconds.")

    while True:
        try:
            logger.info("Starting periodic synchronization check...")
            synchronizer.synchronize()
            
        except Exception as e:
            # Log any unexpected errors during the sync cycle and continue.
            logger.error(f"❌ An unexpected error occurred during the sync cycle: {e}", exc_info=True)
            # In case of a major failure, wait a bit longer before retrying.
            time.sleep(sync_interval_seconds)
        
        # Wait for the next interval.
        time.sleep(sync_interval_seconds)

if __name__ == "__main__":
    main()


