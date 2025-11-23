import albumentations as A
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random, cv2, os
import math

# This assumes your script's working directory has an 'assets' folder.
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

def load_asset_png(name):
    path = os.path.join(ASSETS_DIR, name)
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # RGBA if exists
        if img is None:
            return None
        # convert BGRA->RGBA
        if img.shape[2] == 4:
            b,g,r,a = cv2.split(img)
            rgba = cv2.merge([r,g,b,a])
            return rgba
        else:
            # no alpha channel, convert to RGBA with full alpha
            b,g,r = cv2.split(img)
            a = np.full_like(b, 255)
            rgba = cv2.merge([r,g,b,a])
            return rgba
    return None

GLASSES_PNG = load_asset_png("glasses.png")
MASK_PNG = load_asset_png("mask.png")

def enhance_small_face(img):
    """Enhance small face details using sharpening and contrast adjustment."""
    h, w = img.shape[:2]
    
    # Apply adaptive histogram equalization for better contrast
    if len(img.shape) == 3:
        # Convert to LAB for better processing
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Light sharpening for small faces
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
    sharpened = cv2.filter2D(img, -1, kernel * 0.3)
    img = cv2.addWeighted(img, 0.7, sharpened, 0.3, 0)
    
    return np.clip(img, 0, 255).astype(np.uint8)

def simulate_surveillance_blur(img):
    """Simulate motion blur common in surveillance footage."""
    blur_type = random.choice(['motion', 'gaussian', 'lens'])
    
    if blur_type == 'motion':
        # Simulate motion blur with random direction
        angle = random.uniform(0, 360)
        length = random.randint(3, 8)
        
        # Create motion blur kernel
        kernel_size = length
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        
        # Calculate line points
        cx, cy = kernel_size // 2, kernel_size // 2
        dx = math.cos(math.radians(angle))
        dy = math.sin(math.radians(angle))
        
        for i in range(length):
            x = int(cx + i * dx)
            y = int(cy + i * dy)
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0
        
        kernel = kernel / np.sum(kernel)
        img = cv2.filter2D(img, -1, kernel)
    
    elif blur_type == 'gaussian':
        # Light gaussian blur
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    
    else:  # lens blur
        # Simulate slight lens blur
        img = cv2.blur(img, (3, 3))
    
    return img

def add_surveillance_noise(img):
    """Add realistic surveillance camera noise."""
    noise_type = random.choice(['gaussian', 'salt_pepper', 'poisson'])
    
    if noise_type == 'gaussian':
        # Low-light Gaussian noise
        noise = np.random.normal(0, random.uniform(5, 15), img.shape)
        img = img.astype(np.float32) + noise
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    elif noise_type == 'salt_pepper':
        # Salt and pepper noise (sensor noise)
        prob = random.uniform(0.001, 0.005)
        noise = np.random.random(img.shape[:2])
        img[noise < prob/2] = 0
        img[noise > 1 - prob/2] = 255
    
    else:  # poisson noise
        # Poisson noise (photon noise)
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        img = img.astype(np.float32)
        img = np.random.poisson(img * vals) / float(vals)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    return img

def simulate_angle_variations(img):
    """Simulate different viewing angles common in surveillance."""
    # Random slight perspective transformation
    h, w = img.shape[:2]
    
    # Create subtle perspective distortion points
    margin = min(w, h) * 0.1
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # Random perspective shifts (subtle for faces)
    shift_range = margin * 0.3
    pts2 = np.float32([
        [random.uniform(-shift_range/2, shift_range/2), 
         random.uniform(-shift_range/2, shift_range/2)],
        [w + random.uniform(-shift_range/2, shift_range/2), 
         random.uniform(-shift_range/2, shift_range/2)],
        [random.uniform(-shift_range/2, shift_range/2), 
         h + random.uniform(-shift_range/2, shift_range/2)],
        [w + random.uniform(-shift_range/2, shift_range/2), 
         h + random.uniform(-shift_range/2, shift_range/2)]
    ])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    return img

def random_occlusion_block(img):
    """Optimized occlusion for surveillance scenarios."""
    h, w = img.shape[:2]
    
    # Smaller occlusions for face images
    bw = random.randint(max(2, int(w*0.03)), max(4, int(w*0.15)))
    bh = random.randint(max(2, int(h*0.03)), max(4, int(h*0.12)))
    x = random.randint(0, max(0, w-bw))
    y = random.randint(0, max(0, h-bh))
    
    occlusion_type = random.choice(['dark', 'blur', 'shadow'])
    
    if occlusion_type == 'dark':
        # Dark occlusion (shadow/object)
        color = [random.randint(0, 30)] * 3
        cv2.rectangle(img, (x, y), (x+bw, y+bh), color, -1)
    
    elif occlusion_type == 'blur':
        # Blurred patch
        patch = img[y:y+bh, x:x+bw].copy()
        if patch.size > 0:
            patch = cv2.GaussianBlur(patch, (5, 5), 0)
            img[y:y+bh, x:x+bw] = patch
    
    else:  # shadow
        # Shadow-like occlusion
        patch = img[y:y+bh, x:x+bw].copy()
        if patch.size > 0:
            shadow = patch.astype(np.float32) * random.uniform(0.3, 0.7)
            img[y:y+bh, x:x+bw] = np.clip(shadow, 0, 255).astype(np.uint8)
    
    return img

def overlay_png_on_face(face_rgb, png_rgba, landmarks=None, placement='auto'):
    """
    Optimized overlay for surveillance scenarios.
    """
    if png_rgba is None:
        return face_rgb
    
    fh, fw = face_rgb.shape[:2]
    ol_h, ol_w = png_rgba.shape[:2]

    # Smaller overlays for surveillance faces
    target_w = int(fw * random.uniform(0.4, 0.7))
    scale = target_w / ol_w
    new_w = max(1, int(ol_w * scale))
    new_h = max(1, int(ol_h * scale))
    overlay = cv2.resize(png_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if landmarks is not None and landmarks.shape == (5, 2):
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        lm = landmarks[3]
        rm = landmarks[4]
        
        if overlay.shape[1] >= overlay.shape[0]:  # glasses
            cx = int((left_eye[0] + right_eye[0]) / 2.0)
            cy = int((left_eye[1] + right_eye[1]) / 2.0)
            ox = cx - new_w // 2
            oy = cy - int(new_h * 0.45)
        else:  # mask
            cxm = int((lm[0] + rm[0]) / 2.0)
            cym = int((nose[1] + lm[1] + rm[1]) / 3.0)
            ox = cxm - new_w // 2
            oy = cym - new_h // 2
    else:
        ox = (fw - new_w) // 2
        oy = (fh - new_h) // 2

    ox = max(0, min(fw - new_w, ox))
    oy = max(0, min(fh - new_h, oy))

    # Ensure overlay and ROI have same dimensions
    roi_h = min(new_h, fh - oy)
    roi_w = min(new_w, fw - ox)
    overlay = overlay[:roi_h, :roi_w]
    
    if overlay.shape[2] == 4:
        ol_b, ol_g, ol_r, ol_a = cv2.split(overlay)
        ol_rgb = cv2.merge([ol_r, ol_g, ol_b])
        alpha = (ol_a.astype('float32') / 255.0)[:, :, None]
    else:
        ol_rgb = overlay
        alpha = np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=np.float32)

    roi = face_rgb[oy:oy+roi_h, ox:ox+roi_w].astype('float32')
    
    if roi.shape[:2] != ol_rgb.shape[:2]:
        ol_rgb = cv2.resize(ol_rgb, (roi.shape[1], roi.shape[0]))
        alpha = cv2.resize(alpha, (roi.shape[1], roi.shape[0]))
        if len(alpha.shape) == 2:
            alpha = alpha[:, :, None]

    comp = (alpha * ol_rgb.astype('float32') + (1 - alpha) * roi).astype('uint8')
    face_rgb[oy:oy+roi.shape[0], ox:ox+roi.shape[1]] = comp
    return face_rgb

def get_augs(image_size=112):
    """
    Optimized augmentation pipeline for surveillance face recognition.
    Focuses on small, angled faces under various lighting and noise conditions.
    """
    return A.Compose([
        # Geometric transformations for angled surveillance faces
        A.RandomResizedCrop(
            (image_size, image_size), 
            scale=(0.75, 1.0),  # Wider scale range for small faces
            ratio=(0.85, 1.15), # Allow for slightly different aspect ratios
            p=0.7
        ),
        
        # Horizontal flip for data diversity
        A.HorizontalFlip(p=0.5),
        
        # Rotation for angled faces (common in surveillance)
        A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT, p=0.6),
        
        # Perspective transformation for viewing angle variations
        A.Perspective(scale=(0.02, 0.08), p=0.4),
        
        # Advanced color/lighting augmentations for surveillance scenarios
        A.ColorJitter(
            brightness=0.25,    # Higher brightness variation for lighting changes
            contrast=0.25,      # Higher contrast for surveillance cameras
            saturation=0.2,     # Moderate saturation changes
            hue=0.02,          # Minimal hue changes to maintain face color
            p=0.7
        ),
        
        # Lighting variations (very important for surveillance)
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.6
        ),
        
        # Gamma correction for different lighting conditions
        A.RandomGamma(gamma_limit=(70, 130), p=0.4),
        
        # Shadow simulation
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=3,
            p=0.3
        ),
        
        # Motion blur (very common in surveillance)
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
        ], p=0.4),
        
        # Noise simulation (surveillance camera noise)
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.1, 0.3), p=1.0),
        ], p=0.3),
        
        # Compression artifacts (IP camera compression)
        A.ImageCompression(quality_lower=75, quality_upper=95, p=0.3),
        
        # Partial occlusions
        A.CoarseDropout(
            max_holes=2, 
            max_height=int(image_size*0.12), 
            max_width=int(image_size*0.12), 
            p=0.3
        ),
        
        # Small geometric distortions
        A.ShiftScaleRotate(
            shift_limit=0.08,   # Increased for camera positioning variations
            scale_limit=0.15,   # Scale variations for distance changes
            rotate_limit=15,    # Rotation for head pose variations
            border_mode=cv2.BORDER_REFLECT,
            p=0.6
        ),
        
        # Elastic transform for slight facial expression changes
        A.ElasticTransform(
            alpha=1,
            sigma=10,
            alpha_affine=10,
            p=0.2
        ),
        
        # Grid distortion for lens distortion effects
        A.GridDistortion(num_steps=3, distort_limit=0.1, p=0.2),
    ])

def apply_aug(pil_image, aug, landmarks=None, overlay_prob=0.4):
    """
    Enhanced augmentation application with surveillance-specific enhancements.
    """
    np_img = np.array(pil_image)  # HWC RGB
    
    # Apply base augmentations
    res = aug(image=np_img)['image']
    
    # Additional surveillance-specific augmentations (applied with probability)
    
    # Enhance small face details
    if random.random() < 0.3:
        res = enhance_small_face(res)
    
    # Add surveillance-specific blur
    if random.random() < 0.25:
        res = simulate_surveillance_blur(res)
    
    # Add surveillance camera noise
    if random.random() < 0.2:
        res = add_surveillance_noise(res)
    
    # Simulate angle variations
    if random.random() < 0.3:
        res = simulate_angle_variations(res)
    
    # Distance simulation (zoom variations)
    if random.random() < 0.4:
        # Simulate near/far positioning
        factor = random.uniform(0.8, 1.2)
        h, w = res.shape[:2]
        new_h = max(4, int(h * factor))
        new_w = max(4, int(w * factor))
        
        if factor >= 1.0:
            # Zoom in (closer face)
            resized = cv2.resize(res, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # Center crop back to original size
            cy, cx = new_h//2, new_w//2
            y1 = max(0, cy - h//2)
            x1 = max(0, cx - w//2)
            res = resized[y1:y1+h, x1:x1+w]
            if res.shape[0] != h or res.shape[1] != w:
                res = cv2.resize(res, (w, h))
        else:
            # Zoom out (farther face)
            resized = cv2.resize(res, (new_w, new_h), interpolation=cv2.INTER_AREA)
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            res = cv2.copyMakeBorder(
                resized, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, 
                cv2.BORDER_REFLECT
            )
    
    # Partial occlusion (surveillance-specific)
    if random.random() < 0.2:
        res = random_occlusion_block(res)
    
    # Overlay accessories (glasses/mask) with surveillance-appropriate probability
    if random.random() < overlay_prob:
        choice = random.choice(['glasses', 'mask', 'none', 'none'])  # Lower probability
        if choice == 'glasses' and GLASSES_PNG is not None:
            res = overlay_png_on_face(res, GLASSES_PNG, landmarks)
        elif choice == 'mask' and MASK_PNG is not None:
            res = overlay_png_on_face(res, MASK_PNG, landmarks)
    
    return Image.fromarray(res)
