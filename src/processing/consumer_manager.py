## src/processing/consumer_manager.py:

import subprocess
import time
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ConsumerManager")

# --- MODIFIED: تعداد ورکرها را اینجا تعریف کنید ---
# یک عدد خوب برای شروع، تعداد هسته‌های CPU منهای یک یا دو است.
# با توجه به htop شما، ۶ یا ۸ انتخاب خوبی است.
NUM_WORKERS = 1

def main():
    logger.info(f"🚀 Starting Consumer Manager with {NUM_WORKERS} workers...")
    
    processes = {}

    while True:
        for i in range(NUM_WORKERS):
            worker_id = f"worker-{i+1}"
            
            if worker_id not in processes or processes[worker_id].poll() is not None:
                if worker_id in processes:
                    logger.warning(f"🔄 Worker '{worker_id}' has terminated. Restarting...")
                else:
                    logger.info(f"✨ Starting new worker '{worker_id}'...")
                
                # نیازی به پاس دادن CAMERA_ID نیست چون consumer خودش از پیام می‌خواند
                env = os.environ.copy()
                # env["CAMERA_ID"] = "any" # این دیگر لازم نیست

                cmd = [sys.executable, "src/processing/consumer.py"]

                try:
                    proc = subprocess.Popen(cmd, env=env)
                    processes[worker_id] = proc
                    logger.info(f"👍 Worker '{worker_id}' started successfully with PID {proc.pid}.")
                except Exception as e:
                    logger.error(f"🔥 Failed to start worker '{worker_id}'. Error: {e}")
        
        # هر ۳۰ ثانیه وضعیت ورکرها را چک کن
        time.sleep(30)

if __name__ == "__main__":
    main()