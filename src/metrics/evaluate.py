# # src/metrics/evaluate.py
# import os
# import sys
# import cv2
# import time
# import argparse
# import logging
# from datetime import datetime
# import numpy as np

# # --- بخش جدید: Handler سفارشی برای شستشوی فوری لاگ‌ها ---
# class FlushingFileHandler(logging.FileHandler):
#     def emit(self, record):
#         super().emit(record)
#         self.flush()
# # --- پایان بخش جدید ---

# # افزودن مسیر ریشه پروژه به sys.path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# try:
#     import motmetrics as mm
#     from src.ai.face import FaceProcessor
#     from src.metrics.utils import load_chokepoint_groundtruth_xml, compute_iou
# except ImportError as e:
#     print(f"CRITICAL IMPORT ERROR: {e}. Ensure all dependencies are installed and you're running in the correct environment.")
#     sys.exit(1)

# # --- تنظیمات جدید لاگ ---
# log_dir = "logs"
# os.makedirs(log_dir, exist_ok=True)
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# log_file = os.path.join(log_dir, f"evaluation_{timestamp}.log")

# # پاک کردن handler های قبلی (اگر وجود داشته باشند)
# logging.getLogger().handlers = []

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s | %(levelname)s | %(message)s',
#     # استفاده از handler سفارشی
#     handlers=[
#         FlushingFileHandler(log_file, encoding='utf-8'),
#         logging.StreamHandler(sys.stdout)
#     ]
# )
# logger = logging.getLogger(__name__)

# # --- پوشه جدید برای ذخیره فریم‌های خروجی ---
# output_frames_dir = os.path.join(log_dir, f"output_frames_{timestamp}")
# os.makedirs(output_frames_dir, exist_ok=True)
# # ---

# class Evaluator:
#     def __init__(self, gallery_dir):
#         logger.info("Initializing FaceProcessor for evaluation...")
#         self.face_processor = FaceProcessor(face_db_path=gallery_dir)
#         self.acc = mm.MOTAccumulator(auto_id=True)
#         self.raw_matches = []
#         self.consensus_matches = []
#         self.frame_times = []

#     def process_video(self, video_path, gt_data):
#         logger.info("Starting video processing...")
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             logger.error(f"Failed to open video file: {video_path}")
#             return

#         frame_idx = 0
#         try:
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     logger.info("End of video stream reached.")
#                     break
                
#                 logger.debug(f"Processing frame {frame_idx}...")
#                 start_time = time.perf_counter()
                
#                 processed_frame, results = self.face_processor.process_frame_for_eval(frame)
                
#                 end_time = time.perf_counter()
#                 self.frame_times.append(end_time - start_time)
                
#                 current_gt = gt_data.get(frame_idx, [])
                
#                 # --- بصری‌سازی (بدون تغییر) ---
#                 for gt_item in current_gt:
#                     x1, y1, x2, y2 = map(int, gt_item['bbox'])
#                     person_id = gt_item['id']
#                     cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(processed_frame, f"GT: {person_id}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#                 # ##<-- FIX: ذخیره فریم به صورت تصویر جداگانه -->##
#                 frame_filename = os.path.join(output_frames_dir, f"frame_{frame_idx:05d}.jpg")
#                 cv2.imwrite(frame_filename, processed_frame)
#                 logger.debug(f"Saved frame {frame_idx} to {frame_filename}")

#                 # --- محاسبات متریک (بدون تغییر) ---
#                 gt_boxes = [item['bbox'] for item in current_gt]
#                 gt_ids = [item['id'] for item in current_gt]
#                 pred_boxes = [res['bbox'] for res in results]
#                 pred_track_ids = [res['track_id'] for res in results]
                
#                 distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
#                 self.acc.update(gt_ids, pred_track_ids, distances)
#                 self.collect_recognition_data(results, current_gt)

#                 frame_idx += 1
#                 if frame_idx % 100 == 0:
#                     logger.info(f"Processed {frame_idx} frames...")
        
#         finally:
#             logger.info("Cleaning up video capture resource.")
#             if cap:
#                 cap.release()
#             self.frame_count = frame_idx # ذخیره تعداد فریم‌های پردازش شده

#     # ... (متدهای collect_recognition_data و run_evaluation بدون تغییر) ...
#     def collect_recognition_data(self, pred_results, gt_items):
#         if not pred_results or not gt_items:
#             return

#         for pred in pred_results:
#             best_iou = 0; true_id = None
#             for gt in gt_items:
#                 iou = compute_iou(pred['bbox'], gt['bbox'])
#                 if iou > best_iou:
#                     best_iou, true_id = iou, gt['id']
            
#             if best_iou > 0.5 and true_id:
#                 pred_raw = pred['raw_label']; pred_final = pred['final_label']
#                 if pred_raw and pred_raw != 'Unknown':
#                     self.raw_matches.append({'pred': pred_raw.zfill(4), 'true': true_id})
#                 if pred_final and pred_final != 'Unknown':
#                     self.consensus_matches.append({'pred': pred_final.zfill(4), 'true': true_id})

#     def run_evaluation(self):
#         logger.info("\n" + "="*50); logger.info(" FINAL EVALUATION RESULTS"); logger.info("="*50)
#         mh = mm.metrics.create()
#         summary = mh.compute(self.acc, metrics=['mota', 'idf1'], name='acc')
#         mota = summary['mota']['acc'] * 100
#         idf1 = summary['idf1']['acc'] * 100
#         logger.info(f"[Tracking] MOTA: {mota:.2f}%"); logger.info(f"[Tracking] IDF1: {idf1:.2f}%")
#         avg_fps = 1.0 / np.mean(self.frame_times) if self.frame_times else 0
#         logger.info(f"[Performance] Average FPS: {avg_fps:.2f}")
#         correct_consensus = sum(1 for m in self.consensus_matches if m['pred'] == m['true'])
#         total_consensus = len(self.consensus_matches)
#         rank1_acc = (correct_consensus / total_consensus * 100) if total_consensus > 0 else 0
#         logger.info(f"[Recognition] Rank-1 Accuracy (Consensus): {rank1_acc:.2f}% ({correct_consensus}/{total_consensus})")
#         fpr_raw = sum(1 for m in self.raw_matches if m['pred'] != m['true']) / len(self.raw_matches) if self.raw_matches else 0
#         fpr_cons = sum(1 for m in self.consensus_matches if m['pred'] != m['true']) / len(self.consensus_matches) if self.consensus_matches else 0
#         fpr_reduction = (fpr_raw - fpr_cons) / fpr_raw * 100 if fpr_raw > 0 else 0
#         logger.info(f"[Recognition] FPR Reduction: {fpr_reduction:.2f}% (Raw: {fpr_raw:.4f} -> Consensus: {fpr_cons:.4f})")
#         logger.info("="*50 + "\n")

# def main():
#     parser = argparse.ArgumentParser(description="Evaluate the Stateful Face Recognition System.")
#     parser.add_argument('--video_path', required=True); parser.add_argument('--gallery_dir', required=True)
#     args = parser.parse_args()

#     video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
#     gt_path = os.path.join(os.path.dirname(args.video_path), '../groundtruth', f"{video_basename}.xml")

#     if not os.path.exists(gt_path):
#         logger.error(f"GT file not found: {gt_path}"); sys.exit(1)
    
#     logger.info(f"Video: {args.video_path}"); logger.info(f"GT: {gt_path}"); logger.info(f"Gallery: {args.gallery_dir}")

#     gt_data = load_chokepoint_groundtruth_xml(gt_path)
#     if not gt_data:
#         logger.error("Failed to load GT data."); sys.exit(1)

#     evaluator = Evaluator(gallery_dir=args.gallery_dir)
#     evaluator.process_video(args.video_path, gt_data)
#     evaluator.run_evaluation()

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         logger.exception("A critical error occurred:")
#         sys.exit(1)



# src/metrics/evaluate.py
import os
import sys
import cv2
import time
import argparse
import logging
from datetime import datetime
import numpy as np

# --- Handler سفارشی برای شستشوی فوری لاگ‌ها ---
class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

# افزودن مسیر ریشه پروژه به sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    import motmetrics as mm
    from src.ai.face import FaceProcessor
    from src.metrics.utils import load_chokepoint_groundtruth_xml, compute_iou
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}. Ensure all dependencies are installed and you're running in the correct environment.")
    sys.exit(1)

# --- متغیرهای سراسری برای لاگ و پوشه خروجی ---
# این متغیرها در تابع main مقداردهی خواهند شد
logger = logging.getLogger(__name__)
output_frames_dir = None
# ---

class Evaluator:
    # ... (کلاس Evaluator و تمام متدهای آن بدون هیچ تغییری باقی می‌مانند) ...
    def __init__(self, gallery_dir):
        logger.info("Initializing FaceProcessor for evaluation...")
        self.face_processor = FaceProcessor(face_db_path=gallery_dir)
        self.acc = mm.MOTAccumulator(auto_id=True)
        self.raw_matches = []
        self.consensus_matches = []
        self.frame_times = []

    def process_video(self, video_path, gt_data):
        logger.info("Starting video processing...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream reached.")
                    break
                
                logger.debug(f"Processing frame {frame_idx}...")
                start_time = time.perf_counter()
                
                processed_frame, results = self.face_processor.process_frame_for_eval(frame)
                
                end_time = time.perf_counter()
                self.frame_times.append(end_time - start_time)
                
                current_gt = gt_data.get(frame_idx, [])
                
                for gt_item in current_gt:
                    x1, y1, x2, y2 = map(int, gt_item['bbox'])
                    person_id = gt_item['id']
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"GT: {person_id}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # از متغیر سراسری output_frames_dir استفاده می‌شود
                frame_filename = os.path.join(output_frames_dir, f"frame_{frame_idx:05d}.jpg")
                cv2.imwrite(frame_filename, processed_frame)
                logger.debug(f"Saved frame {frame_idx} to {frame_filename}")

                gt_boxes = [item['bbox'] for item in current_gt]
                gt_ids = [item['id'] for item in current_gt]
                pred_boxes = [res['bbox'] for res in results]
                pred_track_ids = [res['track_id'] for res in results]
                
                distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
                self.acc.update(gt_ids, pred_track_ids, distances)
                self.collect_recognition_data(results, current_gt)

                frame_idx += 1
                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx} frames...")
        
        finally:
            logger.info("Cleaning up video capture resource.")
            if cap:
                cap.release()
            self.frame_count = frame_idx

    def collect_recognition_data(self, pred_results, gt_items):
        if not pred_results or not gt_items: return
        for pred in pred_results:
            best_iou = 0; true_id = None
            for gt in gt_items:
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou: best_iou, true_id = iou, gt['id']
            if best_iou > 0.5 and true_id:
                pred_raw, pred_final = pred['raw_label'], pred['final_label']
                if pred_raw and pred_raw != 'Unknown': self.raw_matches.append({'pred': pred_raw.zfill(4), 'true': true_id})
                if pred_final and pred_final != 'Unknown': self.consensus_matches.append({'pred': pred_final.zfill(4), 'true': true_id})

    def run_evaluation(self):
        logger.info("\n" + "="*50); logger.info(" FINAL EVALUATION RESULTS"); logger.info("="*50)
        mh = mm.metrics.create()
        summary = mh.compute(self.acc, metrics=['mota', 'idf1'], name='acc')
        mota, idf1 = summary['mota']['acc'] * 100, summary['idf1']['acc'] * 100
        logger.info(f"[Tracking] MOTA: {mota:.2f}%"); logger.info(f"[Tracking] IDF1: {idf1:.2f}%")
        avg_fps = 1.0 / np.mean(self.frame_times) if self.frame_times else 0
        logger.info(f"[Performance] Average FPS: {avg_fps:.2f}")
        correct_consensus = sum(1 for m in self.consensus_matches if m['pred'] == m['true'])
        total_consensus = len(self.consensus_matches)
        rank1_acc = (correct_consensus / total_consensus * 100) if total_consensus > 0 else 0
        logger.info(f"[Recognition] Rank-1 Accuracy (Consensus): {rank1_acc:.2f}% ({correct_consensus}/{total_consensus})")
        fpr_raw = sum(1 for m in self.raw_matches if m['pred'] != m['true']) / len(self.raw_matches) if self.raw_matches else 0
        fpr_cons = sum(1 for m in self.consensus_matches if m['pred'] != m['true']) / len(self.consensus_matches) if self.consensus_matches else 0
        fpr_reduction = (fpr_raw - fpr_cons) / fpr_raw * 100 if fpr_raw > 0 else 0
        logger.info(f"[Recognition] FPR Reduction: {fpr_reduction:.2f}% (Raw: {fpr_raw:.4f} -> Consensus: {fpr_cons:.4f})")
        logger.info("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Stateful Face Recognition System.")
    parser.add_argument('--video_path', required=True); parser.add_argument('--gallery_dir', required=True)
    args = parser.parse_args()

    # ##<-- FIX: تمام منطق نام‌گذاری به ابتدای main منتقل شده است -->##
    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    
    # تعریف پوشه اصلی لاگ‌ها
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 1. تعریف مسیر فایل لاگ بر اساس نام ویدیو
    log_file = os.path.join(log_dir, f"{video_basename}.log")
    
    # 2. تعریف پوشه خروجی فریم‌ها بر اساس نام ویدیو
    # از متغیر سراسری استفاده می‌کنیم تا در کلاس Evaluator قابل دسترس باشد
    global output_frames_dir
    output_frames_dir = os.path.join(log_dir, f"{video_basename}")
    os.makedirs(output_frames_dir, exist_ok=True)

    # 3. پیکربندی مجدد لاگ با مسیر فایل جدید
    logging.getLogger().handlers = [] # پاک کردن handler های قبلی
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            FlushingFileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # ##<-- END OF FIX -->##

    gt_path = os.path.join(os.path.dirname(args.video_path), '../groundtruth', f"{video_basename}.xml")

    if not os.path.exists(gt_path):
        logger.error(f"GT file not found: {gt_path}"); sys.exit(1)
    
    logger.info(f"Video: {args.video_path}"); logger.info(f"GT: {gt_path}"); logger.info(f"Gallery: {args.gallery_dir}")

    gt_data = load_chokepoint_groundtruth_xml(gt_path)
    if not gt_data:
        logger.error("Failed to load GT data."); sys.exit(1)

    evaluator = Evaluator(gallery_dir=args.gallery_dir)
    evaluator.process_video(args.video_path, gt_data)
    evaluator.run_evaluation()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("A critical error occurred:")
        sys.exit(1)