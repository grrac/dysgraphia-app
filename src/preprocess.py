"""
Preprocessing for the dysgraphia screening app.

These two functions are copied VERBATIM from FYP2_Preprocessing_Combined.ipynb
(Cell 3). Do not "improve" them — inference must reproduce training exactly,
or the ResNet-18 features will drift away from what scaler.pkl / svm.pkl expect.

Training built Track A as:  resize_and_pad(remove_ruling_lines(line))  (per line,
then 3 lines stacked). The app treats a single uploaded image as the "combined"
sample and runs the same two steps on it.
"""
import cv2
import numpy as np

# Must match deploy_config.json -> input.canvas_h / canvas_w
CANVAS_H, CANVAS_W = 300, 700


def remove_ruling_lines(img):
    """Detect long near-horizontal lines (Hough) and inpaint them; keeps strokes."""
    # 1) isolate candidate horizontal structure
    kernel_w = max(img.shape[1] // 4, 50)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    horiz    = cv2.morphologyEx(img, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # 2) Hough to confirm they are actual long straight lines
    mask = np.zeros_like(img)
    lines = cv2.HoughLinesP(horiz, 1, np.pi / 180, threshold=80,
                            minLineLength=img.shape[1] // 7, maxLineGap=20)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(y2 - y1) <= 3:                      # near-horizontal only
                cv2.line(mask, (x1, y1), (x2, y2), 255, 3)

    # 3) inpaint so text crossing the line is reconstructed, not gouged
    cleaned = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return cleaned


def resize_and_pad(img, canvas_h=CANVAS_H, canvas_w=CANVAS_W):
    h, w  = img.shape
    new_w = max(int(w * 0.4), 1)
    new_h = max(int(h * 0.3), 1)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Safety: if still larger than canvas in either dimension, scale to fit
    rh, rw = resized.shape
    if rh > canvas_h or rw > canvas_w:
        scale   = min(canvas_h / rh, canvas_w / rw)
        resized = cv2.resize(resized, (max(int(rw * scale), 1), max(int(rh * scale), 1)),
                             interpolation=cv2.INTER_AREA)

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)   # black canvas
    rh, rw = resized.shape
    y = (canvas_h - rh) // 2
    x = (canvas_w - rw) // 2
    canvas[y:y + rh, x:x + rw] = resized
    return canvas


def preprocess_single(gray):
    """
    gray : uint8 grayscale numpy array of the uploaded handwriting sample.
    returns : (300, 700) uint8 canvas, ready for ResNet-18 feature extraction.
    """
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    cleaned = remove_ruling_lines(gray)
    return resize_and_pad(cleaned)

def normalize_photo(gray):
    """
    Convert an ordinary ink-on-paper photo (dark strokes, light paper) into the
    dataset format: white strokes on a black background.

    APPROXIMATE — use only for the 'Phone photo' branch. Never call this on
    dataset-format samples (they are already white-on-black).
    """
    gray = cv2.medianBlur(gray, 3)                       # calm phone-camera noise
    binary = cv2.adaptiveThreshold(                      # handles uneven lighting
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,                           # ink -> white, paper -> black
        blockSize=35, C=15,                              # tune if too noisy / too faint
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)  # drop speckles
    return binary