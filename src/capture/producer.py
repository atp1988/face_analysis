## src/capture/producer.py:
import redis
import time
import cv2
import numpy as np
import logging
import pickle
import uuid  # --- MODIFIED: Import uuid for generating unique frame IDs ---
from config.config import settings
from datetime import datetime
import threading
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This client is specific to the producer threads
redis_client = redis.Redis(host='redis', port=6379, db=0, password=None, decode_responses=False)

def camera_producer(camera_id, stop_event):
    rtsp_url = settings.CAMERAS.get(camera_id, {}).get("rtsp")
    if not rtsp_url:
        logger.error(f"RTSP URL for camera_id '{camera_id}' not found.")
        return

    logger.info(f"Starting camera producer for {camera_id}")
    
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error(f"Could not open RTSP stream for {camera_id}")
        return

    FPS_LIMIT = 20
    last_capture_time = 0
    # stream_name = f"camera_stream_{camera_id}"
    stream_name = "camera_processing_tasks"
    frame_counter = 0
    
    try:
        while not stop_event.is_set():
            current_time = time.time()
            if (current_time - last_capture_time) < (1.0 / FPS_LIMIT):
                time.sleep(0.01)
                continue
            
            last_capture_time = current_time
            
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Could not read frame from {camera_id}. Reconnecting...")
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                continue

            frame_counter += 1

            # --- MODIFIED SECTION START ---

            # 1. Encode the frame to JPEG format
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_bytes = buffer.tobytes()

            # 2. Generate a unique ID for this frame and store the frame data in a temporary Redis key with a 10-second TTL
            frame_id = str(uuid.uuid4())
            redis_key = f"frame:{frame_id}"
            redis_client.setex(redis_key, 10, frame_bytes)

            # 3. Create a lightweight task message with the frame ID instead of the full frame data
            task_message = {
                'frame_id': frame_id,
                'camera_id': camera_id,
                'timestamp_producer_utc': datetime.now(timezone.utc).isoformat(),
                'frame_count': frame_counter 
            }
            
            try:
                # 4. Add the lightweight message to the stream and cap the stream size to ~100 entries
                redis_client.xadd(stream_name, {'data': pickle.dumps(task_message)}, maxlen=100, approximate=True)
                logger.debug(f"📤 Frame from {camera_id} (Count: {frame_counter}, ID: {frame_id}) sent to Redis stream '{stream_name}'")
            
            # --- MODIFIED SECTION END ---
            except Exception as e:
                logger.error(f"Failed to send frame to Redis for {camera_id}: {e}")
                time.sleep(5)

    except Exception as e:
        logger.error(f"Unhandled error in camera producer for {camera_id}: {e}", exc_info=True)
    finally:
        cap.release()
        logger.info(f"Camera producer for {camera_id} has stopped.")

def video_producer(video_path, processor_id, stop_event=None):
    """Process video frames using OpenCV and send to Redis."""
    try:
        logger.info(f"Starting video_producer for video {processor_id}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video {video_path}")
            return

        frame_count = 0
        while cap.isOpened() and (stop_event is None or not stop_event.is_set()):
            ret, frame = cap.read()
            if not ret:
                break
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            compressed_frame = buffer.tobytes()
            timestamp = time.time()
            frame_data = {'frame': compressed_frame, 'timestamp': timestamp, 'processor_id': processor_id}
            try:
                # --- MODIFIED: Added MAXLEN to video producer as well for consistency ---
                redis_client.xadd(f"frame_stream_{processor_id}", {'data': pickle.dumps(frame_data)}, maxlen=100, approximate=True)
                logger.info(f"Video {processor_id} - Frame {frame_count} sent to Redis")
            except Exception as e:
                logger.error(f"Failed to send frame to Redis for video {processor_id}: {e}")
            frame_count += 1
            time.sleep(0.5)  # Simulate processing delay
    except Exception as e:
        logger.error(f"Error in video producer for {processor_id}: {e}", exc_info=True)
    finally:
        cap.release()
        if stop_event:
            stop_event.set()
        logger.info(f"Video {processor_id} - Processing complete")