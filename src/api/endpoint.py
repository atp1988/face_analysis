from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from typing import List, Dict
import cv2
import numpy as np
from ai.face import FaceProcessor
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from capture.producer import camera_producer
import threading
import logging
import shutil
import base64
import time
from pydantic import BaseModel
import asyncio
import redis.asyncio as redis
from config.config import settings
import pickle
import json # <-- ADDED: For saving results metadata
from datetime import datetime, timezone

# --- Connection Manager for WebSockets (No changes here) ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, camera_id: str):
        await websocket.accept()
        if camera_id not in self.active_connections:
            self.active_connections[camera_id] = []
        self.active_connections[camera_id].append(websocket)
        logger.info(f"New WebSocket connection for camera '{camera_id}'. Total connections for this camera: {len(self.active_connections[camera_id])}")

    def disconnect(self, websocket: WebSocket, camera_id: str):
        if camera_id in self.active_connections:
            self.active_connections[camera_id].remove(websocket)
            logger.info(f"WebSocket disconnected for camera '{camera_id}'. Remaining connections: {len(self.active_connections[camera_id])}")

    async def broadcast_to_camera(self, camera_id: str, message: bytes):
        if camera_id in self.active_connections:
            for connection in self.active_connections[camera_id]:
                await connection.send_bytes(message)

manager = ConnectionManager()

class RegisterPersonRequest(BaseModel):
    images: List[str]
    person_name: str

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

logger.debug("Initializing FaceProcessor in main process")
try:
    face_processor = FaceProcessor()
except Exception as e:
    logger.error(f"Failed to initialize FaceProcessor: {e}", exc_info=True)
    raise

camera_threads = {}
stop_events = {}

async def redis_frame_listener(manager: ConnectionManager):
    redis_client = redis.from_url(f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}", encoding="utf-8", decode_responses=False)
    pubsub = redis_client.pubsub()
    await pubsub.psubscribe(f"stream_output_*")
    logger.info("Redis listener subscribed to 'stream_output_*' channels for live streaming.")
    while True:
        try:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message["type"] == "pmessage":
                try:
                    payload = pickle.loads(message['data'])
                    frame_data = payload['frame_bytes']
                    processed_ts_str = payload['processed_ts_utc']
                    processed_dt = datetime.fromisoformat(processed_ts_str)
                    api_reception_delay = (datetime.now(timezone.utc) - processed_dt).total_seconds()
                    logger.info(f"🚀 Final delivery delay (Process End -> API): {api_reception_delay:.4f}s")
                    channel_name = message['channel'].decode('utf-8')
                    camera_id = channel_name.split('_')[-1]
                    await manager.broadcast_to_camera(camera_id, frame_data)
                except Exception as e:
                     logger.error(f"Could not process published message: {e}")
        except Exception as e:
            logger.error(f"Error in Redis listener: {e}", exc_info=True)
            await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Starting Redis frame listener...")
    asyncio.create_task(redis_frame_listener(manager))

def get_iran_timestamp():
    return datetime.now(tz=ZoneInfo("Asia/Tehran")).strftime("%Y%m%d_%H%M%S_%f")

@app.websocket("/ws/stream/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    await manager.connect(websocket, camera_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, camera_id)

@app.post("/image")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        processed_frame, personnel_ids = face_processor.process_frame(frame, camera_id="image_upload")
        if processed_frame is None:
            raise HTTPException(status_code=500, detail="Failed to process image")
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_image = buffer.tobytes()
        output_dir = "/app/outputs/images"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = get_iran_timestamp()
        output_path = os.path.join(output_dir, f"processed_{timestamp}.jpg")
        with open(output_path, "wb") as f:
            f.write(processed_image)
        logger.info(f"Processed image saved to {output_path}")
        return {
            "personnel_ids": personnel_ids,
            "processed_image": base64.b64encode(processed_image).decode('utf-8')
        }
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ##<-- NEW: This function will run in the background -->##
def process_video_in_background(temp_video_path: str, output_dir: str):
    """
    Reads a video from a temporary path, processes each frame, saves the results,
    and cleans up the temporary file.
    """
    logger.info(f"BACKGROUND TASK: Starting video processing for {temp_video_path}")
    cap = None
    try:
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            logger.error(f"BACKGROUND TASK FAILED: Could not open video file {temp_video_path}")
            return

        metadata_results = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, personnel_ids = face_processor.process_frame(frame, camera_id="video_upload")
            if processed_frame is None:
                continue

            # Save the processed frame image
            _, buffer = cv2.imencode('.jpg', processed_frame)
            output_path = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
            with open(output_path, "wb") as f:
                f.write(buffer.tobytes())

            # Store metadata (who was detected)
            metadata_results.append({
                "frame": frame_count,
                "timestamp": get_iran_timestamp(),
                "detected_persons": personnel_ids,
            })
            frame_count += 1
        
        # Save all metadata to a single JSON file at the end
        metadata_path = os.path.join(output_dir, "results.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata_results, f, indent=4)
            
        logger.info(f"BACKGROUND TASK: Successfully processed {frame_count} frames. Results saved in {output_dir}")

    except Exception as e:
        logger.error(f"BACKGROUND TASK FAILED: Error processing video {temp_video_path}: {e}", exc_info=True)
    finally:
        if cap:
            cap.release()
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            logger.info(f"BACKGROUND TASK: Cleaned up temporary file {temp_video_path}")


# ##<-- MODIFIED: The /video endpoint now uses BackgroundTasks -->##
@app.post("/video")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # 1. Create unique names for temp file and output directory
        timestamp = get_iran_timestamp()
        video_name = file.filename.rsplit(".", 1)[0]
        temp_video_path = f"/tmp/{video_name}_{timestamp}.mp4"
        output_dir = f"/app/outputs/videos/{video_name}_{timestamp}"
        
        # 2. Save the uploaded video file quickly
        with open(temp_video_path, "wb") as f:
            f.write(await file.read())
            
        # 3. Create the directory where results will be stored
        os.makedirs(output_dir, exist_ok=True)

        # 4. Add the heavy processing task to the background
        background_tasks.add_task(
            process_video_in_background,
            temp_video_path=temp_video_path,
            output_dir=output_dir
        )

        # 5. Return an immediate response to the user
        logger.info(f"Video '{file.filename}' accepted. Processing will continue in the background. Output will be in '{output_dir}'.")
        return {
            "message": "Video accepted and is being processed in the background.",
            "output_directory": output_dir
        }
    except Exception as e:
        logger.error(f"Error accepting video upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/rtsp/start")
async def start_rtsp():
    # ... (no changes) ...
    from config.config import settings
    for camera_id in settings.CAMERAS.keys():
        if camera_id in camera_threads: continue
        stop_events[camera_id] = threading.Event()
        t = threading.Thread(target=camera_producer, args=(camera_id, stop_events[camera_id]))
        t.daemon = True
        camera_threads[camera_id] = t
        t.start()
        time.sleep(1)
        if not t.is_alive():
            del camera_threads[camera_id]
            del stop_events[camera_id]
    return {"message": "RTSP streams started..."}

@app.post("/rtsp/stop")
async def stop_rtsp():
    # ... (no changes) ...
    for camera_id, stop_event in list(stop_events.items()):
        stop_event.set()
        thread = camera_threads.get(camera_id)
        if thread:
            thread.join(timeout=5)
            del camera_threads[camera_id]
        if camera_id in stop_events:
            del stop_events[camera_id]
    return {"message": "RTSP streams stopped"}

def process_person_in_background(person_name: str, saved_image_paths: list, target_dir: str):
    # ... (no changes) ...
    try:
        logger.info(f"BACKGROUND TASK: Starting incremental database update for '{person_name}'.")
        face_processor.add_person(person_name, saved_image_paths)        
        logger.info(f"BACKGROUND TASK: Successfully completed update for '{person_name}'.")
    except ValueError as e:
        shutil.rmtree(target_dir)
        logger.error(f"BACKGROUND TASK FAILED: Could not add person '{person_name}' to database: {e}", exc_info=True)
    except Exception as e:
        shutil.rmtree(target_dir)
        logger.error(f"BACKGROUND TASK FAILED: Unexpected error while adding person '{person_name}': {e}", exc_info=True)

@app.post("/personnel/upload")
async def register_person(data: RegisterPersonRequest, background_tasks: BackgroundTasks):
    # ... (no changes) ...
    DATABASE_ROOT = "/app/src/ai/face_database"
    person_dir_name = data.person_name
    target_dir = os.path.join(DATABASE_ROOT, person_dir_name)
    if os.path.exists(target_dir):
        logger.warning(f"Directory for '{person_dir_name}' exists. It will be removed and updated.")
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    saved_image_paths = []
    try:
        for idx, image_base64 in enumerate(data.images):
            try:
                if "," in image_base64: image_base64 = image_base64.split(",")[1]
                image_data = base64.b64decode(image_base64)
                file_path = os.path.join(target_dir, f"image_{idx+1}.jpg")
                with open(file_path, "wb") as f: f.write(image_data)
                saved_image_paths.append(file_path)
            except Exception as e:
                shutil.rmtree(target_dir)
                raise HTTPException(status_code=500, detail=f"Error saving image {idx+1}: {str(e)}")
    except Exception as e:
        shutil.rmtree(target_dir)
        raise HTTPException(status_code=500, detail="Failed to save uploaded images.")
    background_tasks.add_task(
        process_person_in_background,
        data.person_name,
        saved_image_paths,
        target_dir,
    )
    return {"message": f"Personnel '{person_dir_name}' accepted for processing."}

@app.delete("/personnel/delete/{person_name}")
async def delete_person_directly(person_name: str):
    # ... (no changes) ...
    logger.info(f"Direct request to delete person with name: {person_name}")
    DB_PICKLE_PATH = "/app/src/ai/face_db.pkl"
    DATABASE_ROOT = "/app/src/ai/face_database"
    pkl_deleted = False
    dir_deleted = False
    try:
        if not os.path.exists(DB_PICKLE_PATH):
            raise HTTPException(status_code=500, detail="Database file (face_db.pkl) not found.")
        with open(DB_PICKLE_PATH, "rb") as f: known_faces_data = pickle.load(f)
        if person_name not in known_faces_data:
            raise HTTPException(status_code=404, detail=f"Person with name '{person_name}' not found in the database.")
        del known_faces_data[person_name]
        with open(DB_PICKLE_PATH, "wb") as f: pickle.dump(known_faces_data, f)
        pkl_deleted = True
        logger.info(f"Successfully updated {DB_PICKLE_PATH} by removing '{person_name}'.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error while handling the database file: {str(e)}")
    try:
        person_dir_to_delete = os.path.join(DATABASE_ROOT, person_name)
        if os.path.isdir(person_dir_to_delete):
            shutil.rmtree(person_dir_to_delete)
            dir_deleted = True
            logger.info(f"Image directory '{person_dir_to_delete}' deleted successfully.")
        else:
            logger.warning(f"Image directory for person '{person_name}' was not found.")
    except Exception as e:
        return {
            "message": f"Warning: Person '{person_name}' was deleted from the database, but their image directory could not be removed.",
            "details": {"person_name": person_name, "deleted_from_pkl": pkl_deleted, "directory_deleted": False, "error": f"Error deleting directory: {str(e)}"}
        }
    return {
        "message": f"Person '{person_name}' was successfully deleted.",
        "details": {"person_name": person_name, "deleted_from_pkl": pkl_deleted, "directory_deleted": dir_deleted}
    }