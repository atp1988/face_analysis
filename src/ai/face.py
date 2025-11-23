'''the version of the face.py uses FAISS for fast searching on face embeddings.'''
import cv2
import numpy as np
import os
import pickle
import logging
import traceback
from PIL import Image
from insightface.app import FaceAnalysis
from sklearn.cluster import KMeans
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import deque
import faiss
from tqdm import tqdm

from boxmot.tracker_zoo import create_tracker

from .augment import get_augs, apply_aug

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ai.face")

def l2_normalize(x, eps=1e-10):
    """L2-normalizes a vector."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x)
    return x / (norm + 1e-6) if norm > eps else x

def calculate_iou(box1, box2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

class TrackedFace:
    """A helper class to store state information for each tracked face."""
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox
        self.label = None
        self.recent_labels = deque(maxlen=5)
        self.age = 0
        self.frames_since_recognition = 0

class FaceProcessor:
    """
    A stateful FaceProcessor integrating InsightFace with BoxMOT and Faiss
    for robust, fast, and consistent face tracking and recognition.
    """
    def __init__(self,
                 model_path='/app/src/ai',
                 face_db_path='/app/src/ai/face_database',
                 pickle_path='/app/src/ai/face_db.pkl',
                 det_size=(640, 640),
                 device='cpu',
                 metric='euclidean',
                 tracker_type='bytetrack'):
        logger.info(f"Initializing Stateful FaceProcessor (ArcFace + BoxMOT/{tracker_type})")
        self.model_path = model_path
        self.face_db_path = face_db_path
        self.pickle_path = pickle_path
        self.det_size = det_size
        self.device = device
        self.metric = metric.lower() if metric.lower() in ['cosine', 'euclidean'] else 'cosine'
        
        self.detector = FaceAnalysis(name='buffalo_l', root=model_path, allowed_modules=['detection', 'recognition'])
        ctx_id = 0 if device != 'cpu' else -1
        self.detector.prepare(ctx_id=ctx_id, det_size=self.det_size)

        self.match_threshold = 1.0 if self.metric == 'euclidean' else 0.45
        self.ratio_thresh = 0.85
        self.conf_thresh = 0.5

        self.RECOGNITION_INTERVAL = 5
        self.STALE_TRACK_THRESHOLD = 30
        self.CONFIRMATION_COUNT = 3
        self.IOU_MATCHING_THRESHOLD = 0.6

        self.tracker = create_tracker(tracker_type, tracker_config=os.path.join(model_path, 'models', 'bytetrack.yml'), device=device, half=False, per_class=False)
        self.tracked_faces = {}
        
        self.face_db = {}
        self.faiss_index = None
        self.index_to_pid_map = []
        
        self._load_face_database()
        logger.info("FaceProcessor initialized with %d identities.", len(self.face_db))
        self.pickle_last_modified = 0.0 
        
    def _save_database_to_pickle(self):
        try:
            with open(self.pickle_path, 'wb') as f: pickle.dump(self.face_db, f)
            logger.info("Saved updated face_db pickle to %s with %d identities.", self.pickle_path, len(self.face_db))
        except Exception as e: logger.exception("Failed to save face database pickle: %s", e)

    def _rebuild_faiss_index(self):
        logger.info("Rebuilding Faiss index for fast recognition...")
        if not self.face_db:
            self.faiss_index, self.index_to_pid_map = None, []
            logger.warning("Face database is empty. Faiss index is cleared.")
            return

        all_prototypes, self.index_to_pid_map = [], []
        for pid, data in self.face_db.items():
            for proto in data['prototypes']:
                all_prototypes.append(proto); self.index_to_pid_map.append(pid)
        
        if not all_prototypes: self.faiss_index = None; return

        all_prototypes_np = np.asarray(all_prototypes, dtype=np.float32)
        embedding_dim = all_prototypes_np.shape[1]
        
        index_metric = faiss.METRIC_L2 if self.metric == 'euclidean' else faiss.METRIC_INNER_PRODUCT
        self.faiss_index = faiss.IndexFlat(embedding_dim, index_metric)

        if self.metric == 'cosine': faiss.normalize_L2(all_prototypes_np)

        self.faiss_index.add(all_prototypes_np)
        logger.info(f"Successfully rebuilt Faiss index with {self.faiss_index.ntotal} prototypes.")

    def _load_face_database(self):
        if os.path.exists(self.pickle_path):
            try:
                with open(self.pickle_path, 'rb') as f: self.face_db = pickle.load(f)
                logger.info("Loaded face database from pickle with %d persons.", len(self.face_db))
                self._rebuild_faiss_index()
                return
            except Exception as e: logger.warning("Could not load pickle file: %s. Rebuilding database.", e)
        
        logger.info("Pickle not found. Building new augmented face database...")
        if not os.path.exists(self.face_db_path): os.makedirs(self.face_db_path)
        aug_pipeline = get_augs(image_size=112)
        temp_person_embs = {}
        person_folders = [p for p in os.listdir(self.face_db_path) if os.path.isdir(os.path.join(self.face_db_path, p))]
        for person_id in person_folders:
            logger.info(f"Processing {person_id}:")
            person_dir = os.path.join(self.face_db_path, person_id)
            image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_steps = len(image_files) * 41  # 1 original + 40 augs per image
            person_embeddings = []
            with tqdm(total=total_steps, desc=person_id, unit='emb') as pbar:
                for img_file in image_files:
                    try:
                        pil_img = Image.open(os.path.join(person_dir, img_file)).convert("RGB")
                        # ##<-- FIX 1: Corrected typo from _ to 2 -->##
                        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) 
                        faces = self.detector.get(cv_img)
                        if faces:
                            emb = getattr(max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1])), 'normed_embedding', None)
                            if emb is not None: person_embeddings.append(emb)
                        pbar.update(1)  # Update for original

                        for _ in range(40):
                            aug_pil = apply_aug(pil_img, aug_pipeline)
                            # ##<-- FIX 2: Corrected typo from _ to 2 -->##
                            aug_cv = cv2.cvtColor(np.array(aug_pil), cv2.COLOR_RGB2BGR)
                            faces = self.detector.get(aug_cv)
                            if faces:
                                emb = getattr(max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1])), 'normed_embedding', None)
                                if emb is not None: person_embeddings.append(emb)
                            pbar.update(1)  # Update for each aug
                    except Exception as e:
                        logger.exception(f"Could not process image {img_file} for {person_id}: {e}")
            if person_embeddings: temp_person_embs[person_id] = np.array(person_embeddings, dtype=np.float32)

        self.face_db = {}
        for pid, embs in temp_person_embs.items():
            if not embs.any(): continue
            k = min(5, len(embs));
            if k < 1: continue
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embs)
            prototypes = [l2_normalize(center) for center in kmeans.cluster_centers_]
            self.face_db[pid] = {'prototypes': prototypes, 'n_examples': len(embs)}
        self._save_database_to_pickle()
        self._rebuild_faiss_index()

    def check_and_reload_db_if_changed(self):
        try:
            if not os.path.exists(self.pickle_path):
                if self.face_db: 
                    logger.warning("Pickle file seems to be deleted. Clearing in-memory database.")
                    self.face_db = {}
                    self.pickle_last_modified = 0.0
                return
            current_mtime = os.path.getmtime(self.pickle_path)
            if current_mtime > self.pickle_last_modified:
                logger.info("Change detected in face_db.pkl. Reloading database...")
                with open(self.pickle_path, 'rb') as f:
                    self.face_db = pickle.load(f)
                self.pickle_last_modified = current_mtime
                self._rebuild_faiss_index()
                logger.info("Successfully reloaded face database. Now tracking %d identities.", len(self.face_db))
        except Exception as e:
            logger.error("Failed to check or reload face database pickle: %s", e, exc_info=True)

    def add_person(self, person_name: str, image_paths: list):
        if person_name in self.face_db: raise ValueError(f"Person with name '{person_name}' already exists.")
        logger.info(f"Adding new person: {person_name}")
        aug_pipeline = get_augs(image_size=112)
        person_embeddings = []
        total_steps = len(image_paths) * 41
        with tqdm(total=total_steps, desc=person_name, unit='emb') as pbar:
            for img_path in image_paths:
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    # ##<-- FIX 3: Corrected typo from _ to 2 -->##
                    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    faces = self.detector.get(cv_img)
                    if faces:
                        emb = getattr(max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1])), 'normed_embedding', None)
                        if emb is not None: person_embeddings.append(emb)
                    pbar.update(1)

                    for _ in range(40):
                        aug_pil = apply_aug(pil_img, aug_pipeline)
                        # ##<-- FIX 4: Corrected typo from _ to 2 -->##
                        aug_cv = cv2.cvtColor(np.array(aug_pil), cv2.COLOR_RGB2BGR)
                        faces = self.detector.get(aug_cv)
                        if faces:
                            emb = getattr(max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1])), 'normed_embedding', None)
                            if emb is not None: person_embeddings.append(emb)
                        pbar.update(1)
                except Exception as e:
                    logger.exception(f"Could not process image {img_path} for {person_name}: {e}")
        if not person_embeddings:
            logger.warning(f"No valid embeddings found for {person_name}. Skipping addition.")
            return
        embs_np = np.array(person_embeddings, dtype=np.float32)
        k = min(5, len(embs_np))
        if k < 1: return
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embs_np)
        prototypes = [l2_normalize(center) for center in kmeans.cluster_centers_]
        self.face_db[person_name] = {'prototypes': prototypes, 'n_examples': len(embs_np)}
        self._save_database_to_pickle()
        self._rebuild_faiss_index()
        logger.info(f"Successfully added {person_name} with {len(embs_np)} embeddings.")

    def reload_database(self):
        logger.info("Full database reload requested.")
        if os.path.exists(self.pickle_path):
            try: os.remove(self.pickle_path)
            except Exception as e: logger.error(f"Could not remove pickle file for reload: {e}")
        self._load_face_database()

    def _match_one_to_one(self, embeddings_list):
        labels = ["Unknown"] * len(embeddings_list)
        if not embeddings_list or self.faiss_index is None or self.faiss_index.ntotal == 0: return labels
        try:
            query_embeddings = np.asarray(embeddings_list, dtype=np.float32)
            if self.metric == 'cosine': faiss.normalize_L2(query_embeddings)
            k = min(2, self.faiss_index.ntotal)
            distances, indices = self.faiss_index.search(query_embeddings, k)
            candidates = []
            for i in range(len(query_embeddings)):
                best_idx, best_dist = indices[i][0], distances[i][0]
                if (self.metric == 'euclidean' and best_dist > self.match_threshold) or \
                   (self.metric == 'cosine' and best_dist < self.match_threshold): continue
                if k > 1 and self.index_to_pid_map[best_idx] != self.index_to_pid_map[indices[i][1]]:
                    second_best_dist = distances[i][1]
                    if (self.metric == 'euclidean' and best_dist > second_best_dist * self.ratio_thresh) or \
                       (self.metric == 'cosine' and best_dist < second_best_dist + (1 - self.ratio_thresh)): continue
                candidates.append((i, self.index_to_pid_map[best_idx], best_dist))
            candidates.sort(key=lambda x: x[2], reverse=(self.metric == 'cosine'))
            assigned_pids, assigned_dets = set(), set()
            for det_i, pid, score in candidates:
                if det_i in assigned_dets or pid in assigned_pids: continue
                labels[det_i] = pid
                assigned_pids.add(pid); assigned_dets.add(det_i)
            return labels
        except Exception as e:
            logger.exception("Error in Faiss-based matching: %s", e)
            return ["Unknown"] * len(embeddings_list)

    def process_frame(self, frame, camera_id="default_cam"):
        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed to process_frame")
            return frame, []
        try:
            all_detected_faces = self.detector.get(frame)
            high_conf_faces = [face for face in all_detected_faces if face.det_score >= self.conf_thresh] if all_detected_faces else []
            
            detections_for_tracker = np.array([list(face.bbox) + [face.det_score, 0] for face in high_conf_faces]) if high_conf_faces else np.empty((0, 6))
            online_targets_np = self.tracker.update(detections_for_tracker, frame)
            
            current_track_ids, embeddings_to_match, ordered_track_ids_for_recog = set(), [], []
            if online_targets_np.size > 0:
                for target in online_targets_np:
                    track_bbox, track_id = target[:4], int(target[4])
                    current_track_ids.add(track_id)
                    if track_id not in self.tracked_faces: self.tracked_faces[track_id] = TrackedFace(track_id, track_bbox)
                    track = self.tracked_faces[track_id]
                    track.bbox, track.age, track.frames_since_recognition = track_bbox, 0, track.frames_since_recognition + 1
                    if track.label is None or track.frames_since_recognition >= self.RECOGNITION_INTERVAL:
                        ious = [calculate_iou(track_bbox, det.bbox) for det in high_conf_faces]
                        if not ious: continue
                        best_match_idx = np.argmax(ious)
                        if ious[best_match_idx] >= self.IOU_MATCHING_THRESHOLD and high_conf_faces[best_match_idx].normed_embedding is not None:
                            embeddings_to_match.append(high_conf_faces[best_match_idx].normed_embedding)
                            ordered_track_ids_for_recog.append(track_id)
            
            if embeddings_to_match:
                raw_labels = self._match_one_to_one(embeddings_to_match)
                for i, track_id in enumerate(ordered_track_ids_for_recog):
                    track = self.tracked_faces[track_id]
                    track.frames_since_recognition = 0
                    track.recent_labels.append(raw_labels[i])
                    majority_label = max(set(track.recent_labels), key=list(track.recent_labels).count)
                    if majority_label != "Unknown" and list(track.recent_labels).count(majority_label) >= self.CONFIRMATION_COUNT:
                        if track.label != majority_label: logger.info(f"Track ID {track_id} confirmed as -> {majority_label}")
                        track.label = majority_label

            stale_ids = [tid for tid in self.tracked_faces if tid not in current_track_ids]
            for track_id in stale_ids:
                self.tracked_faces[track_id].age += 1
                if self.tracked_faces[track_id].age > self.STALE_TRACK_THRESHOLD: del self.tracked_faces[track_id]

            annotated_frame_for_return = frame.copy()
            
            for track_id, track in self.tracked_faces.items():
                if track_id not in current_track_ids: continue
                x1, y1, x2, y2 = map(int, track.bbox)
                display_label = track.label if track.label else "Unknown"
                color = (0, 255, 0) if track.label and track.label != "Unknown" else (0, 0, 255)
                cv2.rectangle(annotated_frame_for_return, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame_for_return, f"ID-{track_id}: {display_label}", (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            final_labels_on_frame = [t.label if t.label else "Processing..." for tid, t in self.tracked_faces.items() if tid in current_track_ids]
            return annotated_frame_for_return, final_labels_on_frame

        except Exception as e:
            logger.exception("Error in stateful process_frame: %s\n%s", e, traceback.format_exc())
            return frame, []

    def process_frame_for_eval(self, frame):
        """
        یک نسخه از process_frame که خروجی دقیق‌تری برای ارزیابی برمی‌گرداند.
        """
        # این متد تقریباً کپی process_frame است اما خروجی آن متفاوت است.
        # می‌توانید منطق را کپی کرده یا process_frame را طوری بازنویسی کنید که هر دو حالت را پشتیبانی کند.
        # برای سادگی، منطق را اینجا تکرار می‌کنیم.
        
        try:
            # 1. تشخیص چهره
            all_detected_faces = self.detector.get(frame)
            high_conf_faces = [face for face in all_detected_faces if face.det_score >= self.conf_thresh] if all_detected_faces else []

            # 2. به‌روزرسانی ردیاب
            detections_for_tracker = np.array([list(face.bbox) + [face.det_score, 0] for face in high_conf_faces]) if high_conf_faces else np.empty((0, 6))
            online_targets_np = self.tracker.update(detections_for_tracker, frame)

            # 3. مدیریت ترک‌ها و آماده‌سازی برای تشخیص
            current_track_ids = set()
            if online_targets_np.size > 0:
                for target in online_targets_np:
                    track_bbox, track_id = target[:4], int(target[4])
                    current_track_ids.add(track_id)
                    if track_id not in self.tracked_faces:
                        self.tracked_faces[track_id] = TrackedFace(track_id, track_bbox)
                    
                    track = self.tracked_faces[track_id]
                    track.bbox = track_bbox
                    track.age = 0
                    track.frames_since_recognition += 1

            # 4. تشخیص دسته‌ای و به‌روزرسانی ترک‌ها
            embeddings_to_match = []
            tracks_for_recog = []
            for track_id in current_track_ids:
                track = self.tracked_faces[track_id]
                if track.label is None or track.frames_since_recognition >= self.RECOGNITION_INTERVAL:
                    ious = [calculate_iou(track.bbox, det.bbox) for det in high_conf_faces]
                    if not ious: continue
                    best_match_idx = np.argmax(ious)
                    if ious[best_match_idx] >= self.IOU_MATCHING_THRESHOLD and high_conf_faces[best_match_idx].normed_embedding is not None:
                        embeddings_to_match.append(high_conf_faces[best_match_idx].normed_embedding)
                        tracks_for_recog.append(track)

            if embeddings_to_match:
                raw_labels = self._match_one_to_one(embeddings_to_match)
                for i, track in enumerate(tracks_for_recog):
                    raw_label = raw_labels[i]
                    # ذخیره برچسب خام برای ارزیابی FPR Reduction
                    if not hasattr(track, 'raw_label_this_frame'):
                        track.raw_label_this_frame = None
                    if raw_label != "Unknown":
                        track.raw_label_this_frame = raw_label

                    track.frames_since_recognition = 0
                    track.recent_labels.append(raw_label)
                    majority_label = max(set(track.recent_labels), key=list(track.recent_labels).count)
                    if majority_label != "Unknown" and list(track.recent_labels).count(majority_label) >= self.CONFIRMATION_COUNT:
                        track.label = majority_label
            
            # 5. پاکسازی ترک‌های قدیمی
            stale_ids = [tid for tid in self.tracked_faces if tid not in current_track_ids]
            for track_id in stale_ids:
                self.tracked_faces[track_id].age += 1
                if self.tracked_faces[track_id].age > self.STALE_TRACK_THRESHOLD:
                    del self.tracked_faces[track_id]
            
            # 6. آماده‌سازی خروجی برای ارزیابی
            results_for_eval = []
            annotated_frame = frame.copy()
            for track_id in current_track_ids:
                track = self.tracked_faces[track_id]
                x1, y1, x2, y2 = map(int, track.bbox)
                
                # --- بخش مهم خروجی ---
                results_for_eval.append({
                    'bbox': track.bbox.tolist(),
                    'track_id': track_id,
                    'raw_label': getattr(track, 'raw_label_this_frame', None),
                    'final_label': track.label,
                })
                # پاک کردن برچسب خام برای فریم بعدی
                if hasattr(track, 'raw_label_this_frame'):
                    track.raw_label_this_frame = None

                # رسم روی فریم (اختیاری)
                display_label = track.label if track.label else "Processing..."
                color = (0, 255, 0) if track.label else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"ID-{track_id}: {display_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
            return annotated_frame, results_for_eval

        except Exception:
            # در صورت خطا، یک خروجی خالی برگردان
            return frame, []