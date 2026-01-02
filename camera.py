"""
VideoCamera: The Brain of the Football Player Tracking System
Handles YOLO11 detection, ByteTrack tracking, and Adaptive OCR for jersey numbers.
"""

import cv2
import numpy as np
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, List
from ultralytics import YOLO
import supervision as sv
from PIL import Image


class PlayerState(Enum):
    HUNTING = "HUNTING"
    LOCKED = "LOCKED"
    LOST = "LOST"  # Player not currently visible


class TeamType(Enum):
    UNKNOWN = "UNKNOWN"
    TEAM_RED = "TEAM_RED"      # White numbers on Red jersey
    TEAM_TURQUOISE = "TEAM_TURQUOISE"  # Black numbers on Turquoise jersey


@dataclass
class PlayerTrack:
    """Tracks state and jersey number for a single player."""
    player_id: int  # Fixed ID 1-8 (NOT tracker_id from ByteTrack)
    tracker_id: Optional[int] = None  # Current ByteTrack ID (can change)
    state: PlayerState = PlayerState.HUNTING
    jersey_number: Optional[str] = None
    confidence: float = 0.0
    detection_count: int = 0  # Count of same number detections
    last_seen_frame: int = 0
    frames_since_ocr: int = 0
    frames_since_verification: int = 0
    number_candidates: Dict[str, int] = field(default_factory=dict)
    dominant_color: Tuple[int, int, int] = (128, 128, 128)
    team: TeamType = TeamType.UNKNOWN
    is_visible: bool = False  # Currently visible in frame
    last_bbox: Optional[Tuple[float, float, float, float]] = None  # Last known position (x1, y1, x2, y2)

    def reset_to_hunting(self):
        """Reset player to hunting state."""
        self.state = PlayerState.HUNTING
        self.jersey_number = None
        self.confidence = 0.0
        self.detection_count = 0
        self.number_candidates.clear()

    def mark_lost(self, bbox: Optional[Tuple[float, float, float, float]] = None):
        """Mark player as lost (not visible). Store last known position for spatial matching."""
        self.is_visible = False
        self.tracker_id = None  # Detach from ByteTrack ID
        if bbox is not None:
            self.last_bbox = bbox

    def mark_visible(self, tracker_id: int, frame: int):
        """Mark player as visible with new tracker ID."""
        self.is_visible = True
        self.tracker_id = tracker_id
        self.last_seen_frame = frame
        # If was lost, always go to HUNTING first to verify identity via OCR
        # Even if player has a jersey number, we need to confirm spatial match was correct
        if self.state == PlayerState.LOST:
            self.state = PlayerState.HUNTING
            self.detection_count = 0
            self.confidence = 0.0  # Reset confidence when re-verifying
            self.number_candidates.clear()
            if self.jersey_number is not None:
                print(f"Player P{self.player_id} (#{self.jersey_number}) re-detected, verifying...")


class VideoCamera:
    """
    Main video processing class with YOLO11 detection, ByteTrack tracking,
    and adaptive OCR for jersey number recognition.
    """

    # OCR frequency constants
    HUNTING_OCR_INTERVAL = 2  # Run OCR every 2nd frame when hunting (more aggressive)
    VERIFICATION_INTERVAL = 50  # Verify locked players every 50 frames
    LOCK_THRESHOLD = 3  # Number of consistent detections to lock
    CONFIDENCE_THRESHOLD = 0.5  # Minimum OCR confidence (lowered for preprocessed images)
    LOST_TRACK_TIMEOUT = 35  # Frames before marking player as lost (just above ByteTrack's 30 frame buffer)

    # Inference optimization
    INFERENCE_SIZE = 640  # Resize for YOLO inference
    FRAME_SKIP = 2  # Process every Nth frame for detection

    # Player constraints
    MAX_PLAYERS = 8  # Maximum number of players on the field
    GREEN_HSV_LOWER = (35, 40, 40)  # Lower bound for green field detection
    GREEN_HSV_UPPER = (85, 255, 255)  # Upper bound for green field detection

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.current_frame = 0
        self.is_playing = True
        self.lock = threading.Lock()

        # Fixed 8 player slots (P1-P8) - IDs never change!
        # P1-P4 = RED team, P5-P8 = TURQUOISE team
        self.players: Dict[int, PlayerTrack] = {}
        for i in range(1, self.MAX_PLAYERS + 1):
            player = PlayerTrack(player_id=i)
            # Pre-assign team based on slot - this is PERMANENT
            if i <= 4:
                player.team = TeamType.TEAM_RED
            else:
                player.team = TeamType.TEAM_TURQUOISE
            self.players[i] = player

        # Map ByteTrack tracker_id -> our fixed player_id
        self.tracker_to_player: Dict[int, int] = {}

        self.last_detections = None

        # Initialize YOLO model
        print("Loading YOLO11l model...")
        self.model = YOLO("yolo11l.pt")

        # Initialize ByteTrack tracker via supervision
        # Use stable default settings - don't over-tune
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,  # Default-ish, not too aggressive
            lost_track_buffer=30,             # Keep tracks for 1 second at 30fps
            minimum_matching_threshold=0.8,   # Higher = stricter matching, less ID switches
            frame_rate=int(self.fps)
        )

        # Initialize PaddleOCR (lazy loading to save memory)
        self.ocr = None
        self._init_ocr()

        # Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.TOP_LEFT,
            text_thickness=2,
            text_scale=0.6
        )

        print(f"Video loaded: {self.total_frames} frames at {self.fps:.1f} FPS")

    def _init_ocr(self):
        """Initialize OCR for jersey number recognition."""
        import os
        self.debug_dir = "ocr_debug"
        os.makedirs(self.debug_dir, exist_ok=True)
        self.ocr_attempt_count = 0

        # Try PARSeq first (state-of-the-art, fast, accurate for short text)
        try:
            import torch
            print("Loading PARSeq model...")
            self.parseq_model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True).eval()
            # Get image transform
            from strhub.data.module import SceneTextDataModule
            self.parseq_transform = SceneTextDataModule.get_transform(self.parseq_model.hparams.img_size)
            self.ocr_type = 'parseq'
            # Use GPU if available
            self.parseq_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.parseq_model = self.parseq_model.to(self.parseq_device)
            print(f"PARSeq initialized successfully (device: {self.parseq_device})")
            self.ocr = self.parseq_model  # For compatibility checks
            return
        except ImportError as e:
            print(f"PARSeq not available ({e}), trying EasyOCR...")
        except Exception as e:
            print(f"PARSeq init failed: {e}, trying EasyOCR...")

        # Fallback to EasyOCR
        try:
            import easyocr
            self.ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
            self.ocr_type = 'easyocr'
            print("EasyOCR initialized successfully")
            return
        except ImportError:
            print("EasyOCR not available, trying PaddleOCR...")
        except Exception as e:
            print(f"EasyOCR init failed: {e}, trying PaddleOCR...")

        # Fallback to PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(lang='en')
            self.ocr_type = 'paddleocr'
            print("PaddleOCR initialized successfully")
        except Exception as e:
            print(f"Warning: PaddleOCR initialization failed: {e}")
            self.ocr = None
            self.ocr_type = None

    def get_available_ocr_engines(self) -> Dict[str, bool]:
        """Get list of available OCR engines and their availability status."""
        available = {}

        # Check PARSeq
        try:
            import torch
            # Check if parseq can be loaded (it may be cached)
            available['parseq'] = True
        except ImportError:
            available['parseq'] = False

        # Check EasyOCR
        try:
            import easyocr
            available['easyocr'] = True
        except ImportError:
            available['easyocr'] = False

        # Check PaddleOCR
        try:
            from paddleocr import PaddleOCR
            available['paddleocr'] = True
        except ImportError:
            available['paddleocr'] = False

        return available

    def switch_ocr(self, ocr_type: str) -> Tuple[bool, str]:
        """
        Switch to a different OCR engine.

        Args:
            ocr_type: One of 'parseq', 'easyocr', 'paddleocr'

        Returns:
            Tuple of (success: bool, message: str)
        """
        if ocr_type == self.ocr_type:
            return True, f"Already using {ocr_type}"

        with self.lock:
            old_type = self.ocr_type

            if ocr_type == 'parseq':
                try:
                    import torch
                    print("Switching to PARSeq...")
                    self.parseq_model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True).eval()
                    from strhub.data.module import SceneTextDataModule
                    self.parseq_transform = SceneTextDataModule.get_transform(self.parseq_model.hparams.img_size)
                    self.parseq_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.parseq_model = self.parseq_model.to(self.parseq_device)
                    self.ocr_type = 'parseq'
                    self.ocr = self.parseq_model
                    print(f"Switched to PARSeq (device: {self.parseq_device})")
                    return True, f"Switched to PARSeq (device: {self.parseq_device})"
                except Exception as e:
                    print(f"Failed to switch to PARSeq: {e}")
                    return False, f"Failed to switch to PARSeq: {e}"

            elif ocr_type == 'easyocr':
                try:
                    import easyocr
                    print("Switching to EasyOCR...")
                    self.ocr = easyocr.Reader(['en'], gpu=False, verbose=False)
                    self.ocr_type = 'easyocr'
                    print("Switched to EasyOCR")
                    return True, "Switched to EasyOCR"
                except Exception as e:
                    print(f"Failed to switch to EasyOCR: {e}")
                    return False, f"Failed to switch to EasyOCR: {e}"

            elif ocr_type == 'paddleocr':
                try:
                    from paddleocr import PaddleOCR
                    print("Switching to PaddleOCR...")
                    self.ocr = PaddleOCR(lang='en')
                    self.ocr_type = 'paddleocr'
                    print("Switched to PaddleOCR")
                    return True, "Switched to PaddleOCR"
                except Exception as e:
                    print(f"Failed to switch to PaddleOCR: {e}")
                    return False, f"Failed to switch to PaddleOCR: {e}"

            else:
                return False, f"Unknown OCR type: {ocr_type}"

    def _is_on_green_field(self, frame: np.ndarray, bbox: np.ndarray) -> bool:
        """Check if detection is on the green playing field."""
        x1, y1, x2, y2 = map(int, bbox)

        # Get bottom center of bounding box (player's feet position)
        foot_x = (x1 + x2) // 2
        foot_y = y2

        # Ensure coordinates are within frame
        h, w = frame.shape[:2]
        foot_x = max(0, min(foot_x, w - 1))
        foot_y = max(0, min(foot_y, h - 1))

        # Sample area around feet (10x10 region)
        sample_y1 = max(0, foot_y - 5)
        sample_y2 = min(h, foot_y + 5)
        sample_x1 = max(0, foot_x - 5)
        sample_x2 = min(w, foot_x + 5)

        if sample_y2 <= sample_y1 or sample_x2 <= sample_x1:
            return True  # Default to accepting if can't sample

        sample = frame[sample_y1:sample_y2, sample_x1:sample_x2]

        # Convert to HSV
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)

        # Check for green color (field)
        mask = cv2.inRange(hsv, self.GREEN_HSV_LOWER, self.GREEN_HSV_UPPER)
        green_ratio = np.sum(mask > 0) / mask.size

        # Accept if at least 30% of the area around feet is green
        return green_ratio > 0.3

    def _extract_dominant_color(self, image: np.ndarray) -> Tuple[int, int, int]:
        """Extract dominant color from jersey region for box coloring."""
        if image.size == 0:
            return (128, 128, 128)

        # Resize for faster processing
        small = cv2.resize(image, (20, 20))

        # Convert to RGB and reshape
        pixels = small.reshape(-1, 3)

        # Use k-means to find dominant color (simplified: just use mean)
        dominant = np.mean(pixels, axis=0).astype(int)

        return tuple(dominant.tolist())

    def _detect_team(self, image: np.ndarray) -> TeamType:
        """Detect which team based on jersey color (Red vs Turquoise)."""
        if image.size == 0:
            return TeamType.UNKNOWN

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red color range (wraps around in HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        # Turquoise/Cyan color range
        lower_turquoise = np.array([75, 50, 50])
        upper_turquoise = np.array([100, 255, 255])

        # Create masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_turquoise = cv2.inRange(hsv, lower_turquoise, upper_turquoise)

        # Count pixels
        red_pixels = cv2.countNonZero(mask_red)
        turquoise_pixels = cv2.countNonZero(mask_turquoise)

        total_pixels = image.shape[0] * image.shape[1]
        red_ratio = red_pixels / total_pixels
        turquoise_ratio = turquoise_pixels / total_pixels

        # Threshold for team detection (at least 10% of jersey should be team color)
        if red_ratio > 0.10 and red_ratio > turquoise_ratio:
            return TeamType.TEAM_RED
        elif turquoise_ratio > 0.10 and turquoise_ratio > red_ratio:
            return TeamType.TEAM_TURQUOISE

        return TeamType.UNKNOWN

    def _preprocess_for_team(self, image: np.ndarray, team: TeamType) -> List[np.ndarray]:
        """
        Create preprocessed versions of the image optimized for each team's jersey colors.
        Returns multiple versions to try OCR on.

        Team 1 (Red): White numbers on Red background
        Team 2 (Turquoise): Black numbers on Turquoise background
        """
        preprocessed = []

        if image is None or image.size == 0:
            return preprocessed

        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if team == TeamType.TEAM_RED:
            # White on Red: Extract white/light pixels
            # Method 1: High value in grayscale (white numbers)
            _, white_thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            preprocessed.append(white_thresh)

            # Method 2: Saturation-based (white has low saturation)
            s_channel = hsv[:, :, 1]
            _, low_sat = cv2.threshold(s_channel, 50, 255, cv2.THRESH_BINARY_INV)
            # Combine with brightness
            v_channel = hsv[:, :, 2]
            _, high_val = cv2.threshold(v_channel, 150, 255, cv2.THRESH_BINARY)
            white_mask = cv2.bitwise_and(low_sat, high_val)
            preprocessed.append(white_mask)

            # Method 3: Adaptive threshold on inverted (for contrast)
            inverted = cv2.bitwise_not(gray)
            adaptive = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
            preprocessed.append(cv2.bitwise_not(adaptive))

        elif team == TeamType.TEAM_TURQUOISE:
            # Black on Turquoise: Extract dark pixels
            # Method 1: Low value in grayscale (black numbers)
            _, black_thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
            preprocessed.append(black_thresh)

            # Method 2: Value channel threshold (black is low V)
            v_channel = hsv[:, :, 2]
            _, low_val = cv2.threshold(v_channel, 100, 255, cv2.THRESH_BINARY_INV)
            preprocessed.append(low_val)

            # Method 3: Adaptive threshold
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
            preprocessed.append(adaptive)

        else:
            # Unknown team: try general approaches
            # Otsu's thresholding
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed.append(otsu)
            preprocessed.append(cv2.bitwise_not(otsu))

            # Adaptive threshold both ways
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
            preprocessed.append(adaptive)
            preprocessed.append(cv2.bitwise_not(adaptive))

        # Apply morphological operations to clean up each result
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = []
        for img in preprocessed:
            # Remove noise
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            # Fill small holes
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            cleaned.append(closed)

        return cleaned

    def _crop_torso(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Crop the full torso region where jersey numbers are located.
        Numbers can be very large covering most of the torso (front and back).
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Ensure valid coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        height = y2 - y1

        # Capture full torso area - numbers are large and cover most of it
        # Skip only the head (top ~8%) and legs (bottom ~40%)
        torso_top = y1 + int(height * 0.08)
        torso_bottom = y1 + int(height * 0.60)

        # Full width - no margin since numbers can be wide
        crop_x1 = x1
        crop_x2 = x2

        crop = frame[torso_top:torso_bottom, crop_x1:crop_x2]

        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        return crop

    def _upscale_for_ocr(self, image: np.ndarray, target_height: int = 100) -> np.ndarray:
        """Upscale image for better OCR recognition."""
        if image is None or image.size == 0:
            return image

        h, w = image.shape[:2]
        if h < 20:
            return image

        scale = target_height / h
        new_w = int(w * scale)

        upscaled = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

        # Enhance contrast
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return enhanced

    def _run_ocr_single(self, image: np.ndarray, save_debug: bool = False, debug_label: str = "") -> Tuple[Optional[str], float]:
        """Run OCR on a single image and extract jersey number."""
        if self.ocr is None or image is None or image.size == 0:
            return None, 0.0

        try:
            # Handle both grayscale and color images
            if len(image.shape) == 2:
                # Grayscale - convert to RGB (3 channel)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                # BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            best_number = None
            best_conf = 0.0

            if save_debug and hasattr(self, 'debug_dir'):
                self.ocr_attempt_count += 1
                debug_path = f"{self.debug_dir}/ocr_{self.ocr_attempt_count:04d}_{debug_label}.jpg"
                cv2.imwrite(debug_path, image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

            # Use appropriate OCR based on type
            if self.ocr_type == 'parseq':
                import torch
                # Convert to PIL Image for PARSeq transform
                pil_image = Image.fromarray(rgb_image)
                # Apply transform and add batch dimension
                img_tensor = self.parseq_transform(pil_image).unsqueeze(0).to(self.parseq_device)

                # Run inference
                with torch.no_grad():
                    logits = self.parseq_model(img_tensor)
                    pred = logits.softmax(-1)
                    labels, confidences = self.parseq_model.tokenizer.decode(pred)

                text = labels[0] if labels else ""
                # PARSeq returns per-character confidence, we take the mean
                conf = float(confidences[0].mean()) if confidences and len(confidences[0]) > 0 else 0.0

                if save_debug:
                    print(f"[OCR Debug] PARSeq: '{text}' (conf: {conf:.3f})")

                # Extract digits from recognized text
                digits = ''.join(c for c in str(text) if c.isdigit())
                if digits and len(digits) <= 2:
                    num = int(digits)
                    if 1 <= num <= 99:
                        best_number = digits
                        best_conf = conf

            elif self.ocr_type == 'easyocr':
                # EasyOCR returns list of (bbox, text, conf)
                results = self.ocr.readtext(rgb_image, detail=1, paragraph=False)

                if save_debug:
                    print(f"[OCR Debug] EasyOCR results: {results}")

                for detection in results:
                    if len(detection) >= 3:
                        bbox, text, conf = detection
                        if save_debug:
                            print(f"  -> Detected: '{text}' (conf: {conf:.3f})")

                        digits = ''.join(c for c in str(text) if c.isdigit())
                        if digits and len(digits) <= 2:
                            num = int(digits)
                            if 1 <= num <= 99 and conf > best_conf:
                                best_number = digits
                                best_conf = conf

            elif self.ocr_type == 'paddleocr':
                # PaddleOCR
                results = self.ocr.ocr(rgb_image)

                if save_debug:
                    print(f"[OCR Debug] PaddleOCR results: {results}")

                if results and isinstance(results, list) and len(results) > 0:
                    result = results[0]

                    # New format: dict with 'rec_texts' and 'rec_scores'
                    if isinstance(result, dict):
                        texts = result.get('rec_texts', [])
                        scores = result.get('rec_scores', [])

                        for text, conf in zip(texts, scores):
                            if save_debug:
                                print(f"  -> Detected: '{text}' (conf: {conf:.3f})")

                            digits = ''.join(c for c in str(text) if c.isdigit())
                            if digits and len(digits) <= 2:
                                num = int(digits)
                                if 1 <= num <= 99 and conf > best_conf:
                                    best_number = digits
                                    best_conf = conf

                    # Old format: list of [box, (text, conf)]
                    elif isinstance(result, list):
                        for line in result:
                            if len(line) >= 2:
                                text, conf = line[1]
                                if save_debug:
                                    print(f"  -> Detected: '{text}' (conf: {conf:.3f})")

                                digits = ''.join(c for c in str(text) if c.isdigit())
                                if digits and len(digits) <= 2:
                                    num = int(digits)
                                    if 1 <= num <= 99 and conf > best_conf:
                                        best_number = digits
                                        best_conf = conf

            return best_number, best_conf

        except Exception as e:
            print(f"[OCR Error] {e}")
            return None, 0.0

    def _run_ocr(self, image: np.ndarray, team: TeamType) -> Tuple[Optional[str], float]:
        """
        Run OCR with team-specific preprocessing.
        Tries multiple preprocessing methods and returns the best result.
        """
        if self.ocr is None or image is None or image.size == 0:
            return None, 0.0

        # Debug mode for first 100 OCR attempts
        debug_mode = hasattr(self, 'ocr_attempt_count') and self.ocr_attempt_count < 100

        # Get team-specific preprocessed versions
        preprocessed_images = self._preprocess_for_team(image, team)

        best_number = None
        best_conf = 0.0

        # Try original image first (upscaled)
        orig_upscaled = image
        if image.shape[0] < 100:
            scale = 100 / image.shape[0]
            orig_upscaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        number, conf = self._run_ocr_single(orig_upscaled, save_debug=debug_mode, debug_label="original")
        if number and conf > best_conf:
            best_number = number
            best_conf = conf

        # Try each preprocessed version
        for idx, prep_img in enumerate(preprocessed_images):
            # Upscale preprocessed image for better OCR
            if prep_img.shape[0] < 100:
                scale = 100 / prep_img.shape[0]
                prep_img = cv2.resize(prep_img, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_CUBIC)

            number, conf = self._run_ocr_single(prep_img, save_debug=debug_mode, debug_label=f"prep{idx}")
            if number and conf > best_conf:
                best_number = number
                best_conf = conf

        if debug_mode and best_number:
            print(f"[OCR] Best result: {best_number} (conf: {best_conf:.3f})")

        return best_number, best_conf

    def _process_player_ocr(self, frame: np.ndarray, player: PlayerTrack,
                            bbox: np.ndarray) -> None:
        """Process OCR for a single player based on their state."""
        player.frames_since_ocr += 1
        player.frames_since_verification += 1

        should_run_ocr = False

        if player.state == PlayerState.HUNTING:
            # Aggressive OCR every few frames
            if player.frames_since_ocr >= self.HUNTING_OCR_INTERVAL:
                should_run_ocr = True
        else:  # LOCKED
            # Verification check every N frames
            if player.frames_since_verification >= self.VERIFICATION_INTERVAL:
                should_run_ocr = True
                player.frames_since_verification = 0

        if not should_run_ocr:
            return

        player.frames_since_ocr = 0

        # Crop and process torso
        torso = self._crop_torso(frame, bbox)
        if torso is None:
            return

        # Update dominant color (for display only)
        player.dominant_color = self._extract_dominant_color(torso)

        # NOTE: Team is FIXED based on slot (P1-P4=RED, P5-P8=TRQ)
        # Never change player.team here!

        # Upscale for OCR
        upscaled = self._upscale_for_ocr(torso)

        # Run OCR with team-specific preprocessing
        number, confidence = self._run_ocr(upscaled, player.team)

        if number and confidence >= self.CONFIDENCE_THRESHOLD:
            if player.state == PlayerState.HUNTING:
                # Re-verification: player has jersey number but is in HUNTING (after being lost)
                if player.jersey_number is not None:
                    if number == player.jersey_number:
                        # Same number detected - spatial match was correct, re-lock immediately
                        player.state = PlayerState.LOCKED
                        player.confidence = confidence
                        print(f"Player P{player.player_id} RE-LOCKED as #{number} (verified)")
                        return
                    else:
                        # Different number detected - this might be wrong spatial match!
                        # Count mismatches to confirm
                        mismatch_key = f"_wrong_{number}"
                        player.number_candidates[mismatch_key] = player.number_candidates.get(mismatch_key, 0) + 1
                        print(f"P{player.player_id} (#{player.jersey_number}) seeing #{number} - mismatch count: {player.number_candidates.get(mismatch_key, 0)}")

                        if player.number_candidates.get(mismatch_key, 0) >= 2:
                            # Confirmed wrong match - find the correct player for this detection
                            # ONLY search within the SAME TEAM
                            correct_player_id = self._find_player_by_jersey(number, player.team)
                            if correct_player_id and correct_player_id != player.player_id:
                                # This tracker is actually tracking another player from same team
                                correct_player = self.players[correct_player_id]
                                print(f"FIXING SWAP: Tracker was on P{player.player_id} (#{player.jersey_number}) "
                                      f"but OCR shows #{number} -> reassigning to P{correct_player_id}")

                                # Move this tracker to the correct player
                                old_tracker = player.tracker_id
                                if old_tracker is not None:
                                    # Disconnect from wrong player
                                    player.tracker_id = None
                                    player.is_visible = False
                                    player.state = PlayerState.LOST
                                    player.number_candidates.clear()

                                    # Connect to correct player (same team)
                                    self.tracker_to_player[old_tracker] = correct_player_id
                                    correct_player.tracker_id = old_tracker
                                    correct_player.is_visible = True
                                    correct_player.state = PlayerState.LOCKED
                                    correct_player.last_seen_frame = self.current_frame
                                    correct_player.last_bbox = player.last_bbox
                                    correct_player.number_candidates.clear()

                                    print(f"P{correct_player_id} (#{correct_player.jersey_number}) now LOCKED")
                            else:
                                # No other player in same team has this number - OCR error, re-lock
                                print(f"P{player.player_id} OCR error? No same-team player has #{number}, re-locking as #{player.jersey_number}")
                                player.state = PlayerState.LOCKED
                                player.number_candidates.clear()
                        return

                # Fresh player (no jersey yet) - check if number belongs to a LOST player
                # ONLY search within the SAME TEAM
                existing_player_id = self._find_player_by_jersey(number, player.team)
                if existing_player_id and existing_player_id != player.player_id:
                    existing_player = self.players[existing_player_id]
                    # If the existing player is LOST, this tracker is actually THEM!
                    # Transfer the tracker to the correct player (same team)
                    if existing_player.state == PlayerState.LOST or not existing_player.is_visible:
                        # Count detections before transferring (need consistency)
                        transfer_key = f"_transfer_{number}"
                        player.number_candidates[transfer_key] = player.number_candidates.get(transfer_key, 0) + 1

                        if player.number_candidates.get(transfer_key, 0) >= 2:
                            print(f"TRANSFER: P{player.player_id} detected #{number} which belongs to LOST P{existing_player_id}")
                            old_tracker = player.tracker_id
                            if old_tracker is not None:
                                # Clear the temporary slot (but keep team - it's fixed)
                                player.tracker_id = None
                                player.is_visible = False
                                player.state = PlayerState.LOST
                                player.jersey_number = None  # Reset - it was temporary
                                # NOTE: Don't reset player.team - it's FIXED based on slot!
                                player.number_candidates.clear()

                                # Transfer to the real owner (same team)
                                self.tracker_to_player[old_tracker] = existing_player_id
                                existing_player.tracker_id = old_tracker
                                existing_player.is_visible = True
                                existing_player.state = PlayerState.LOCKED
                                existing_player.confidence = confidence
                                existing_player.last_seen_frame = self.current_frame
                                existing_player.last_bbox = player.last_bbox
                                existing_player.number_candidates.clear()
                                print(f"P{existing_player_id} (#{existing_player.jersey_number}) now LOCKED via transfer")
                        return
                    else:
                        # Existing player is visible - OCR might have misread
                        return

                # Count consistent detections
                player.number_candidates[number] = player.number_candidates.get(number, 0) + 1

                # Check if any number has enough consistent detections
                for num, count in player.number_candidates.items():
                    if num.startswith("_"):  # Skip internal keys like _transfer_, _wrong_
                        continue
                    if count >= self.LOCK_THRESHOLD:
                        # Before locking, check if another player in SAME TEAM already has this number
                        existing_player_id = self._find_player_by_jersey(num, player.team)

                        if existing_player_id and existing_player_id != player.player_id:
                            # Another player in same team has this number - clear candidates and keep hunting
                            player.number_candidates.clear()
                            break

                        player.jersey_number = num
                        player.confidence = confidence
                        player.state = PlayerState.LOCKED
                        player.detection_count = count
                        team_name = "RED" if player.team == TeamType.TEAM_RED else "TRQ"
                        print(f"Player P{player.player_id} ({team_name}) LOCKED as #{num}")
                        break
            else:  # LOCKED - verification
                if number != player.jersey_number:
                    # Mismatch - be tolerant, only reset after many failures
                    player.detection_count -= 1
                    if player.detection_count <= -5:  # More tolerant
                        print(f"Player P{player.player_id} verification failed, resetting")
                        player.reset_to_hunting()
                else:
                    # Confirmed - boost detection count
                    player.detection_count = min(player.detection_count + 1, 10)
                    player.confidence = max(player.confidence, confidence)

    def _get_available_player_slot(self, team: TeamType) -> Optional[int]:
        """Get an available player slot for a specific team.

        P1-P4 = RED team
        P5-P8 = TURQUOISE team

        Only returns slots that:
        1. Match the requested team
        2. Are not currently visible
        3. Have no jersey number assigned yet
        """
        if team == TeamType.TEAM_RED:
            slot_range = range(1, 5)  # P1-P4
        elif team == TeamType.TEAM_TURQUOISE:
            slot_range = range(5, 9)  # P5-P8
        else:
            # Unknown team - can't assign
            return None

        for player_id in slot_range:
            player = self.players[player_id]
            if not player.is_visible and player.tracker_id is None and player.jersey_number is None:
                return player_id

        return None

    def _find_player_by_jersey(self, jersey_number: str, team: TeamType = None) -> Optional[int]:
        """Find player_id that has this jersey number (locked or lost).

        If team is specified, only search within that team's slots.
        """
        if team == TeamType.TEAM_RED:
            search_range = range(1, 5)
        elif team == TeamType.TEAM_TURQUOISE:
            search_range = range(5, 9)
        else:
            search_range = range(1, 9)

        for player_id in search_range:
            player = self.players[player_id]
            if player.jersey_number == jersey_number:
                return player_id
        return None

    def _find_lost_player_to_reconnect(self) -> Optional[int]:
        """Find a lost player with known jersey who could be reconnected."""
        for player_id in sorted(self.players.keys()):
            player = self.players[player_id]
            if not player.is_visible and player.tracker_id is None and player.jersey_number is not None:
                return player_id
        return None

    def _bbox_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Calculate center point of a bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _bbox_distance(self, bbox1: Tuple[float, float, float, float],
                       bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate Euclidean distance between centers of two bounding boxes."""
        c1 = self._bbox_center(bbox1)
        c2 = self._bbox_center(bbox2)
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    def _assign_tracker_to_player(self, tracker_id: int, bbox: np.ndarray, frame: np.ndarray) -> Optional[int]:
        """
        Assign a ByteTrack tracker_id to one of our fixed player slots.

        IMPORTANT: Team determines slot range:
        - RED team -> P1-P4
        - TURQUOISE team -> P5-P8

        Team is detected from jersey color BEFORE assignment and is PERMANENT.
        """
        # Already assigned?
        if tracker_id in self.tracker_to_player:
            player_id = self.tracker_to_player[tracker_id]
            # Update last_bbox for active players
            self.players[player_id].last_bbox = tuple(bbox)
            return player_id

        # NEW TRACKER: Detect team color FIRST
        torso = self._crop_torso(frame, bbox)
        if torso is None:
            return None

        detected_team = self._detect_team(torso)

        if detected_team == TeamType.UNKNOWN:
            # Can't assign without knowing team - skip this frame
            return None

        # Get available slot for this team
        player_id = self._get_available_player_slot(detected_team)

        if player_id is None:
            return None  # All slots for this team are taken

        # Assign to slot (team is already set in __init__)
        self.tracker_to_player[tracker_id] = player_id
        self.players[player_id].last_bbox = tuple(bbox)
        self.players[player_id].mark_visible(tracker_id, self.current_frame)

        team_name = "RED" if detected_team == TeamType.TEAM_RED else "TRQ"
        print(f"New tracker {tracker_id} assigned to P{player_id} ({team_name})")
        return player_id

    def _try_reconnect_lost_player(self, frame: np.ndarray, tracker_id: int, bbox: np.ndarray) -> bool:
        """
        Try to reconnect an unassigned tracker to a LOST player via OCR.
        Called when no fresh slots are available for this team.

        IMPORTANT: Only reconnects to players of the SAME TEAM.
        P1-P4 = RED, P5-P8 = TURQUOISE

        Returns True if successfully reconnected, False otherwise.
        """
        # Skip if this tracker is somehow already assigned
        if tracker_id in self.tracker_to_player:
            return False

        # Crop torso for team detection
        torso = self._crop_torso(frame, bbox)
        if torso is None:
            return False

        # Detect team color FIRST - this determines which players we can reconnect to
        detected_team = self._detect_team(torso)

        if detected_team == TeamType.UNKNOWN:
            return False  # Can't reconnect without knowing team

        # Only look at LOST players from the SAME TEAM
        if detected_team == TeamType.TEAM_RED:
            team_slot_range = range(1, 5)  # P1-P4
        else:
            team_slot_range = range(5, 9)  # P5-P8

        lost_players = [self.players[pid] for pid in team_slot_range
                       if self.players[pid].state == PlayerState.LOST
                       and self.players[pid].jersey_number is not None]

        if not lost_players:
            return False

        # Upscale and run OCR
        upscaled = self._upscale_for_ocr(torso)
        number, confidence = self._run_ocr(upscaled, detected_team)

        if not number or confidence < self.CONFIDENCE_THRESHOLD:
            return False

        # Find LOST player with this jersey number (same team only)
        for player in lost_players:
            if player.jersey_number == number:
                # Reconnect this tracker to the LOST player
                self.tracker_to_player[tracker_id] = player.player_id
                player.tracker_id = tracker_id
                player.is_visible = True
                player.state = PlayerState.LOCKED
                player.last_seen_frame = self.current_frame
                player.last_bbox = tuple(bbox)
                player.confidence = confidence
                player.number_candidates.clear()

                team_name = "RED" if detected_team == TeamType.TEAM_RED else "TRQ"
                print(f"RECONNECTED: Tracker {tracker_id} -> P{player.player_id} (#{number}, {team_name}) via OCR")
                return True

        return False

    def _update_player_visibility(self, active_tracker_ids: set) -> None:
        """Update visibility status for all players."""
        for player_id, player in self.players.items():
            if player.tracker_id is not None:
                if player.tracker_id not in active_tracker_ids:
                    # This player's tracker is no longer active
                    frames_lost = self.current_frame - player.last_seen_frame
                    # Must be slightly longer than ByteTrack's lost_track_buffer (30 frames)
                    # to ensure ByteTrack has truly dropped the track
                    if frames_lost > 35:
                        # Mark as lost - store last known position
                        old_tracker = player.tracker_id
                        player.mark_lost(player.last_bbox)
                        if old_tracker in self.tracker_to_player:
                            del self.tracker_to_player[old_tracker]
                        if player.state != PlayerState.LOST:
                            player.state = PlayerState.LOST
                            print(f"Player P{player_id} (#{player.jersey_number}) marked as LOST at {player.last_bbox}")

    def get_frame(self) -> Optional[bytes]:
        """
        Get the next processed frame with annotations.
        Returns JPEG-encoded bytes or None if video ended.
        """
        with self.lock:
            if not self.is_playing:
                # Return last frame if paused
                if hasattr(self, '_last_frame_bytes'):
                    return self._last_frame_bytes
                return None

            ret, frame = self.cap.read()

            if not ret:
                # Video ended
                return None

            self.current_frame += 1

            # Skip frames for performance (but still increment counter)
            process_this_frame = (self.current_frame % self.FRAME_SKIP == 0)

            if process_this_frame:
                # Resize for inference
                h, w = frame.shape[:2]
                scale = self.INFERENCE_SIZE / max(h, w)

                if scale < 1:
                    inference_frame = cv2.resize(
                        frame,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_LINEAR
                    )
                else:
                    inference_frame = frame
                    scale = 1.0

                # Run YOLO detection with LOWER confidence to catch distant players
                results = self.model(
                    inference_frame,
                    classes=[0, 32],  # person=0, sports ball=32
                    conf=0.15,  # Lower confidence to catch players far away
                    verbose=False
                )[0]

                # Convert to supervision detections
                detections = sv.Detections.from_ultralytics(results)

                # Scale bboxes back to original size
                if scale < 1:
                    detections.xyxy = detections.xyxy / scale

                # Filter for persons only for tracking
                person_mask = detections.class_id == 0
                person_detections = detections[person_mask]

                # Filter to only players on the green field
                if len(person_detections) > 0:
                    on_field_mask = []
                    for bbox in person_detections.xyxy:
                        on_field_mask.append(self._is_on_green_field(frame, bbox))
                    on_field_mask = np.array(on_field_mask)
                    person_detections = person_detections[on_field_mask]

                # Update tracker
                tracked_detections = self.tracker.update_with_detections(person_detections)

                # Store for annotation
                self.last_detections = tracked_detections

                # STEP 1: Collect all active ByteTrack IDs this frame
                active_tracker_ids = set()
                if tracked_detections.tracker_id is not None:
                    for tracker_id in tracked_detections.tracker_id:
                        if tracker_id is not None:
                            active_tracker_ids.add(tracker_id)

                # STEP 2: Update visibility FIRST - mark lost players before assigning new trackers
                # This is CRITICAL to prevent swapping when ByteTrack reassigns IDs
                self._update_player_visibility(active_tracker_ids)

                # STEP 3: Now assign trackers to player slots and process OCR
                if tracked_detections.tracker_id is not None:
                    for i, tracker_id in enumerate(tracked_detections.tracker_id):
                        if tracker_id is None:
                            continue

                        bbox = tracked_detections.xyxy[i]

                        # Assign this tracker to a fixed player slot (P1-P4 RED, P5-P8 TRQ)
                        player_id = self._assign_tracker_to_player(tracker_id, bbox, frame)

                        if player_id is None:
                            # No fresh slots for this team - try to reconnect via OCR to a LOST player
                            self._try_reconnect_lost_player(frame, tracker_id, bbox)
                            continue

                        player = self.players[player_id]
                        player.last_seen_frame = self.current_frame

                        # Process OCR
                        self._process_player_ocr(frame, player, bbox)

            # Annotate frame
            annotated_frame = self._annotate_frame(frame)

            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()

            self._last_frame_bytes = frame_bytes

            return frame_bytes

    def _annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Annotate frame with player boxes and labels."""
        annotated = frame.copy()

        if self.last_detections is None or self.last_detections.tracker_id is None:
            return annotated

        # Larger font settings for visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2  # Much larger text
        font_thickness = 3
        box_thickness = 3

        for i, tracker_id in enumerate(self.last_detections.tracker_id):
            if tracker_id is None:
                continue

            bbox = self.last_detections.xyxy[i]
            x1, y1, x2, y2 = map(int, bbox)

            # Get our fixed player_id from tracker_id
            player_id = self.tracker_to_player.get(tracker_id)
            if player_id is None:
                continue

            player = self.players.get(player_id)
            if player is None:
                continue

            # Box color based on team
            if player.team == TeamType.TEAM_RED:
                color = (0, 0, 255)  # Red in BGR
            elif player.team == TeamType.TEAM_TURQUOISE:
                color = (208, 224, 64)  # Turquoise in BGR
            else:
                color = (128, 128, 128)  # Gray for unknown

            # Label text and color based on state
            if player.state == PlayerState.LOCKED:
                label = f"P{player_id}: #{player.jersey_number}"
                text_color = (0, 255, 0)  # Green
            else:
                if player.number_candidates:
                    best_guess = max(player.number_candidates.items(),
                                    key=lambda x: x[1])[0]
                    label = f"P{player_id}: {best_guess}?"
                else:
                    label = f"P{player_id}: ?"
                text_color = (0, 0, 255)  # Red

            # Draw bounding box (thicker)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

            # Draw label background (larger)
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            # Background rectangle with padding
            padding = 8
            cv2.rectangle(annotated,
                         (x1, y1 - text_h - padding * 2 - baseline),
                         (x1 + text_w + padding * 2, y1),
                         (0, 0, 0), -1)

            # Draw label text (larger and bolder)
            cv2.putText(annotated, label,
                       (x1 + padding, y1 - padding - baseline),
                       font, font_scale,
                       text_color, font_thickness)

        # Draw frame counter (larger)
        cv2.putText(annotated,
                   f"Frame: {self.current_frame}/{self.total_frames}",
                   (10, 40), font, 1.0,
                   (255, 255, 255), 2)

        return annotated

    def get_stats(self) -> dict:
        """Return current statistics for the UI."""
        with self.lock:
            players_data = []

            # Show all 8 players (P1-P8), including lost ones
            for player_id in sorted(self.players.keys()):
                player = self.players[player_id]

                # Skip players that have never been seen
                if player.last_seen_frame == 0 and not player.is_visible:
                    continue

                # Team display name
                team_name = "?"
                if player.team == TeamType.TEAM_RED:
                    team_name = "Red"
                elif player.team == TeamType.TEAM_TURQUOISE:
                    team_name = "Turquoise"

                # Determine status for display
                if player.is_visible:
                    status = player.state.value
                else:
                    status = "LOST"

                players_data.append({
                    'id': f"P{player_id}",
                    'number': player.jersey_number or '?',
                    'confidence': f"{player.confidence * 100:.0f}%" if player.confidence > 0 else '-',
                    'status': status,
                    'team': team_name
                })

            return {
                'current_frame': self.current_frame,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'is_playing': self.is_playing,
                'players': players_data
            }

    def toggle_play(self) -> bool:
        """Toggle play/pause state. Returns new state."""
        with self.lock:
            self.is_playing = not self.is_playing
            return self.is_playing

    def seek_to_frame(self, frame_number: int) -> bool:
        """
        Seek to a specific frame in the video.

        Args:
            frame_number: The frame to seek to (0-indexed)

        Returns:
            True if seek was successful, False otherwise
        """
        with self.lock:
            # Clamp frame number to valid range
            frame_number = max(0, min(frame_number, self.total_frames - 1))

            # Set video capture position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number

            # Reset ByteTrack tracker - it will re-track from new position
            self.tracker.reset()

            # Clear tracker mappings (trackers will be reassigned)
            self.tracker_to_player.clear()

            # Reset all players to LOST state but keep their jersey numbers
            for player in self.players.values():
                if player.jersey_number is not None:
                    # Player had identity - mark as LOST, keep jersey
                    player.tracker_id = None
                    player.is_visible = False
                    player.state = PlayerState.LOST
                    player.number_candidates.clear()
                else:
                    # Player never had identity - fully reset
                    player.tracker_id = None
                    player.is_visible = False
                    player.state = PlayerState.HUNTING
                    player.number_candidates.clear()

            self.last_detections = None

            print(f"Seeked to frame {frame_number}")
            return True

    def get_single_frame(self) -> Optional[bytes]:
        """
        Get the current frame without advancing.
        Used when paused and seeking to show the frame at the seek position.
        """
        with self.lock:
            # Save current position
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

            # Read frame at current position
            ret, frame = self.cap.read()

            if not ret:
                return None

            # Reset position (so get_frame() will read same frame when resumed)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

            # Annotate frame (without processing - just show current detections)
            annotated_frame = self._annotate_frame(frame)

            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 70])
            return buffer.tobytes()

    def release(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()

    def __del__(self):
        self.release()
