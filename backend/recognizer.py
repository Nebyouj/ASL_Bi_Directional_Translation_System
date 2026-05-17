import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# ==============================
# CONFIGURATION
# ==============================
SEQUENCE_LENGTH = 20
PAUSE_FRAMES = 5
MOTION_THRESHOLD = 0.005
CONFIDENCE_THRESHOLD = 0.5



class ASLRecognizer:
    def __init__(self, model, labels):
        self.model = model
        self.labels = labels


        self.sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.sentence = []

        self.prev_features = None
        self.still_frames = 0
        self.frame_timestamp = 0
        self.last_word = None

        # ------------------------------
        # MediaPipe Setup
        # ------------------------------
        base_options = python.BaseOptions(
            model_asset_path="hand_landmarker.task"
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        print("✅ ASL Recognizer Initialized")

    # ==============================
    # Feature Extraction
    # ==============================
    def extract_features(self, result):
        features = []
        for i in range(2):
            if i < len(result.hand_landmarks):
                lm = result.hand_landmarks[i]
                wrist_x, wrist_y = lm[0].x, lm[0].y  # ✅ wrist origin
                for p in lm:
                    features.append(p.x - wrist_x)
                    features.append(p.y - wrist_y)
            else:
                features.extend([0.0] * 42)
        return features

    # ==============================
    # Motion Detection
    # ==============================
    def motion_detected(self, curr):
        if self.prev_features is None:
            return True, 0.0

        diff = np.mean(
            np.abs(np.array(curr) - np.array(self.prev_features))
        )

        motion = diff > MOTION_THRESHOLD
        return motion, diff
    
    def run_prediction(self):
        x = torch.tensor(
        np.array(list(self.sequence)),
        dtype=torch.float32
        ).unsqueeze(0)


        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, dim=1)

        confidence = confidence.item()
        pred = pred.item()

        word = self.labels[pred]

        print(f"Prediction: {word} | Confidence: {confidence:.3f}")

        if confidence >= 0.3:
            if word != self.last_word:
                self.sentence.append(word)
                self.last_word = word
                print(f"✅ WORD ADDED: {word}")

        self.sequence.clear()
        self.still_frames = 0

        return word


    # ==============================
    # Frame Processing
    # ==============================
    def process_frame(self, frame):

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Ensure contiguous uint8 memory
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # 🔥 Proper timestamp increment (30 FPS approx)
        self.frame_timestamp += 33

        result = self.detector.detect_for_video(
            mp_image,
            self.frame_timestamp
        )

        if not result.hand_landmarks:
            print("❌ No hands detected")

            # # Treat no hands as pause
            # self.still_frames += 1

            # if (
            #     self.still_frames >= PAUSE_FRAMES and
            #     len(self.sequence) >= SEQUENCE_LENGTH
            # ):
            #     print("🟡 Pause from hand removal — running prediction...")
            #     return self.run_prediction()

            return None


        features = self.extract_features(result)

        motion, diff = self.motion_detected(features)

        print(f"Motion diff: {diff:.5f}")

        if motion:
            self.sequence.append(features)
            self.still_frames = 0
        else:
            self.still_frames += 1

        print(
            f"Sequence length: {len(self.sequence)} | "
            f"Still frames: {self.still_frames}"
        )

        self.prev_features = features

        # ==============================
        # WORD BOUNDARY DETECTION
        # ==============================
        if (
            self.still_frames >= PAUSE_FRAMES and
            len(self.sequence) >= SEQUENCE_LENGTH
        ):

            print("🟡 Pause detected — running prediction...")

            x = torch.tensor(
                np.array(list(self.sequence)),
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(x)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred = torch.max(probs, dim=1)

            confidence = confidence.item()
            pred = pred.item()

            word = self.labels[pred]

            print(f"Prediction: {word} | Confidence: {confidence:.3f}")

            # Only accept confident predictions
            if confidence >= CONFIDENCE_THRESHOLD:

                # Prevent repeated same word spam
                if word != self.last_word:
                    self.sentence.append(word)
                    self.last_word = word

                    print(f"✅ WORD ADDED: {word}")
                else:
                    print("⚠️ Same word ignored (duplicate)")
            else:
                print("⚠️ Low confidence — ignored")

            # Reset buffers
            self.sequence.clear()
            self.still_frames = 0

            return word

        return None
