# src/metrics/utils.py
import xml.etree.ElementTree as ET
import numpy as np

def load_chokepoint_groundtruth_xml(gt_path):
    """
    Ground Truth دیتاست ChokePoint را از فایل XML بارگذاری می‌کند.
    از آنجایی که XML فاقد Bounding Box است، یک BBox منطقی بر اساس مختصات
    چشم‌ها تخمین زده می‌شود.
    """
    try:
        tree = ET.parse(gt_path)
        root = tree.getroot()
    except (ET.ParseError, FileNotFoundError):
        # اگر فایل GT وجود نداشت یا خراب بود، دیکشنری خالی برگردان
        return {}

    gt_data = {}
    for frame_elem in root.findall('frame'):
        frame_id = int(frame_elem.get('number'))
        
        # ChokePoint از frame index 1-based استفاده می‌کند، ما آن را به 0-based تبدیل می‌کنیم
        frame_idx = frame_id - 1
        if frame_idx < 0: continue

        gt_data[frame_idx] = []

        for person_elem in frame_elem.findall('person'):
            person_id = person_elem.get('id')
            
            # اطمینان از اینکه شناسه همیشه فرمت ۴ رقمی دارد (e.g., "0003")
            # اگرچه در نمونه شما ۳ رقمی است، ۴ رقمی امن‌تر است
            person_id_formatted = person_id.zfill(4)

            le_elem = person_elem.find('leftEye')
            re_elem = person_elem.find('rightEye')

            if le_elem is not None and re_elem is not None:
                try:
                    le = (int(le_elem.get('x')), int(le_elem.get('y')))
                    re = (int(re_elem.get('x')), int(re_elem.get('y')))
                    
                    # --- تخمین Bounding Box از روی چشم‌ها ---
                    # 1. محاسبه فاصله بین چشم‌ها به عنوان مقیاس
                    eye_dist = np.linalg.norm(np.array(le) - np.array(re))
                    if eye_dist == 0: continue

                    # 2. محاسبه مرکز صورت
                    center_x = (le[0] + re[0]) / 2
                    center_y = (le[1] + re[1]) / 2
                    
                    # 3. تخمین عرض و ارتفاع BBox بر اساس یک ضریب منطقی از فاصله چشم‌ها
                    # این ضرایب به صورت تجربی انتخاب شده‌اند
                    width = eye_dist * 2.5
                    height = eye_dist * 3.5

                    # 4. محاسبه مختصات نهایی BBox
                    x1 = center_x - width / 2
                    y1 = center_y - height / 2.5 # صورت معمولا پایین‌تر از مرکز چشم‌هاست
                    x2 = center_x + width / 2
                    y2 = center_y + height / 1.5

                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    
                    gt_data[frame_idx].append({'bbox': bbox, 'id': person_id_formatted})
                except (ValueError, TypeError):
                    continue # رد کردن تگ‌های ناقص
                    
    return gt_data

def compute_iou(box1, box2):
    """محاسبه Intersection over Union (IoU) بین دو bounding box."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area