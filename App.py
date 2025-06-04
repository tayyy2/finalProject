import cv2
import numpy as np
import os
import json
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque
import time 
import json

def preprocess_image(landmarks):
    try:
        print("Preprocessing input landmarks")
        print(f"Input landmarks type: {type(landmarks)}")
        print(f"Number of landmark sequences: {len(landmarks)}")
        
        if not landmarks or len(landmarks) == 0:
            print("ERROR: No landmarks provided")
            return None
        
        flattened_frames = []
        for frame_landmarks in landmarks:
            if not frame_landmarks:
                print("WARNING: Empty frame landmarks")
                continue
            
            flat_frame = []
            # Use the first detected hand (works for 1 or 2 hands)
            hand_landmarks = frame_landmarks[0]  # Always take the first hand
            for landmark in hand_landmarks:
                flat_frame.extend([landmark[0], landmark[1]])  # x, y only
            
            if flat_frame:
                flattened_frames.append(flat_frame)
        
        if not flattened_frames:
            print("ERROR: No valid frames after processing")
            return None
        
        processed_array = np.array(flattened_frames)  # Shape: (15, 42)
        processed_array = np.expand_dims(processed_array, axis=0)  # Shape: (1, 15, 42)
        
        print("Processed array shape:", processed_array.shape)
        return processed_array
    
    except Exception as e:
        print(f"CRITICAL ERROR in preprocess_image: {e}")
        import traceback
        traceback.print_exc()
        return None

class SignLanguageTranslator:
    def __init__(self, root):
        self.root = root
        self.root.title("ລະບົບແປພາສາມື")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Initialize variables BEFORE any method calls
        self.hand_landmarks_buffer = []  # Initialize this FIRST
        self.MAX_BUFFER_SIZE = 15  # Number of landmark sequences to collect before processing
        
        # Initialize other variables
        self.model = None
        self.label_map = None
        self.cap = None
        self.is_running = False
        self.dataset_path = 'dataset'
        self.prediction_queue = deque(maxlen=10)
        self.sentence = []
        self.last_prediction = None
        self.prediction_cooldown = 0
        
        # Setup MediaPipe hands with error handling
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            print(f"Error initializing MediaPipe Hands: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Initialization Error", f"Failed to initialize MediaPipe: {e}")
        
        # Create UI
        self.create_widgets()
        
        # Load model
        self.load_saved_model()
        
        # Initialize webcam
        self.init_webcam()

    def create_widgets(self):
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Top frame - video and prediction
        top_frame = ttk.Frame(main_container)
        top_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Video frame
        video_frame = ttk.LabelFrame(top_frame, text="ໜ້າຕ່າງກ້ອງ", padding="10")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_label = ttk.Label(video_frame)       
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel (predictions and controls)
        right_panel = ttk.Frame(top_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0), ipadx=20)
        
        # Prediction frame
        pred_frame = ttk.LabelFrame(right_panel, text="ການແປປັດຈຸບັນ", padding="10")
        pred_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
        
        self.pred_label = ttk.Label(pred_frame, text="ຍັງບໍ່ມີການແປ", font=("Arial", 28, "bold"))
        self.pred_label.pack(pady=10)
        
        self.confidence_label = ttk.Label(pred_frame, text="Confidence: 0%", font=("Arial", 12))
        self.confidence_label.pack(pady=5)
        
        # Hand detection info
        self.hands_label = ttk.Label(pred_frame, text="ກວດຈັບມຶ: 0", font=("Arial", 12))
        self.hands_label.pack(pady=5)
        
        # Sentence frame
        sentence_frame = ttk.LabelFrame(right_panel, text="ຄຳກ່ອນໜ້າ", padding="10")
        sentence_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.sentence_text = scrolledtext.ScrolledText(sentence_frame, wrap=tk.WORD, height=6, font=("Arial", 14))
        self.sentence_text.pack(fill=tk.BOTH, expand=True)
        
        # Sentence control buttons
        sentence_buttons = ttk.Frame(sentence_frame)
        sentence_buttons.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(sentence_buttons, text="ຍະວ່າງ", command=self.add_space).pack(side=tk.LEFT, padx=5)
        ttk.Button(sentence_buttons, text="ລຶບຄຳກ່ອນໜ້າ", command=self.delete_last_word).pack(side=tk.LEFT, padx=5)
        ttk.Button(sentence_buttons, text="ລຶບທັງໝົດ", command=self.clear_sentence).pack(side=tk.LEFT, padx=5)
         
        # Bottom frame - controls
        bottom_frame = ttk.Frame(main_container)
        bottom_frame.pack(fill=tk.X, pady=10)
        
        # Status
        status_frame = ttk.LabelFrame(bottom_frame, text="ສະຖານະ", padding="10")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="ສະຖານະ : ຍັງບໍ່ເລີ່ມ")
        self.status_label.pack(fill=tk.X)
        
        # Control buttons
        button_frame = ttk.Frame(main_container)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Add sensitivity slider
        sensitivity_frame = ttk.LabelFrame(button_frame, text="ຄວາມອ່ອນໄຫວຂອງພະຍາກອນ", padding="10")
        sensitivity_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        self.confidence_threshold = tk.DoubleVar(value=70.0)
        self.confidence_scale = ttk.Scale(
            sensitivity_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.confidence_threshold,
            length=150
        )
        self.confidence_scale.pack(side=tk.LEFT)
        
        self.threshold_label = ttk.Label(sensitivity_frame, text="70%")
        self.threshold_label.pack(side=tk.LEFT, padx=(5, 0))
        
        self.confidence_scale.configure(command=self.update_threshold_label)
        
        # Control buttons
        controls = ttk.Frame(button_frame)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.start_button = ttk.Button(controls, text="ເລີ່ມ", command=self.start_recognition)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(controls, text="ຢຸດ", command=self.stop_recognition, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Toggle for hand landmarks
        self.show_landmarks_var = tk.BooleanVar(value=True)
        self.landmarks_check = ttk.Checkbutton(
            controls, 
            text="ສະແດງຂໍ້ຕໍ່ນິ້ວ-ມຶ", 
            variable=self.show_landmarks_var
        )
        self.landmarks_check.pack(side=tk.LEFT, padx=20)
        
        # Add capture button for collecting new examples
        self.capture_var = tk.BooleanVar(value=False)
        self.capture_check = ttk.Checkbutton(
            controls,
            text="Capture Mode",
            variable=self.capture_var
        )
        self.capture_check.pack(side=tk.LEFT, padx=10)
        
        # Add About Us button
        ttk.Button(button_frame, text="ກ່ຽວກັບເຮົາ", command=self.show_about_us).pack(side=tk.LEFT, padx=20)
        
        ttk.Button(button_frame, text="ອອກ", command=self.on_closing).pack(side=tk.RIGHT, padx=5)
        
        # Bind closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        """Handle window closing."""
        self.is_running = False
        
        # Release webcam
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        
        # Release MediaPipe resources
        if hasattr(self, 'hands'):
            self.hands.close()
        
        self.root.destroy()
    
    def show_about_us(self):
        """Show About Us window with team information."""
        about_window = tk.Toplevel(self.root)
        about_window.title("ກ່ຽວກັບເຮົາ")
        about_window.geometry("600x500")
        about_window.resizable(False, False)
        about_window.transient(self.root)
        about_window.grab_set()
        
        # Main frame
        main_frame = ttk.Frame(about_window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="ທີມ", font=("Arial", 18, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Description
        description = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=10)
        description.pack(fill=tk.BOTH, expand=True, pady=10)
        description.insert(tk.END, "ແອັບແປພາສາມຶ\n\n"
                           "Developed as part of a final year project to create an accessible "
                           "communication tool for the deaf and hard of hearing community.")
        description.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(main_frame, text="Close", command=about_window.destroy).pack(pady=10)
    
    def update_threshold_label(self, value):
        """Update the threshold label when slider is moved."""
        threshold = int(float(value))
        self.threshold_label.config(text=f"{threshold}%")
    
    def add_space(self):
        """Add a space to the sentence."""
        self.sentence.append(" ")
        self.update_sentence_display()
    
    def delete_last_word(self):
        """Delete the last word from the sentence."""
        if self.sentence:
            self.sentence.pop()
            self.update_sentence_display()
    
    def clear_sentence(self):
        """Clear the entire sentence."""
        self.sentence = []
        self.update_sentence_display()
    
    def update_sentence_display(self):
        """Update the sentence display text."""
        self.sentence_text.delete(1.0, tk.END)
        sentence_str = "".join(self.sentence)
        self.sentence_text.insert(tk.END, sentence_str)
    
    def load_saved_model(self):
        """Load the trained model and label mapping."""
        try:
            # Check if model exists
            model_path = os.path.join('models', 'lstm_sign_video_model.h5')
            label_map_path = os.path.join('models', 'label_map.json')
            
            if not os.path.exists(model_path):
                self.update_status("Error: Model file not found. Please train the model first.")
                return False
            
            if not os.path.exists(label_map_path):
                self.update_status("Error: Label mapping file not found. Please train the model first.")
                return False
            
            # Load model
            self.update_status("Loading model...")
            
            # Add custom optimizer
            custom_objects = {
                'Adam': tf.keras.optimizers.Adam
            }
            
            self.model = load_model(model_path, custom_objects=custom_objects)
            print("Model summary:")
            self.model.summary()  # Print model architecture
            print("Input shape:", self.model.input_shape)
            
            # Load label mapping
            with open(label_map_path, 'r') as f:
                str_label_map = json.load(f)
                # Convert string keys back to integers
                self.label_map = {int(k): v for k, v in str_label_map.items()}
            
            self.update_status("ໂຫລດຕົວແບບສຳເລັດ")
            return True
            
        except Exception as e:
            self.update_status(f"Error loading model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            return False
    
    def init_webcam(self):
        """Initialize webcam capture."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.update_status("Error: Could not open webcam")
                messagebox.showerror("Error", "Could not open webcam")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.update_status("Webcam initialized")
            return True
            
        except Exception as e:
            self.update_status(f"Error initializing webcam: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize webcam: {str(e)}")
            return False
    
    def update_status(self, message):
        """Update status label."""
        self.status_label.config(text=f"Status: {message}")
        self.root.update_idletasks()

    def capture_hand_data(self, landmarks):
        """
        Capture hand landmark data for potential model training.
        
        Args:
            landmarks (list): List of hand landmarks detected
        """
        try:
            # Ensure dataset directory exists
            os.makedirs(self.dataset_path, exist_ok=True)
            
            # Prompt for sign/class
            sign_dialog = tk.Toplevel(self.root)
            sign_dialog.title("Capture Sign")
            sign_dialog.geometry("300x200")
            
            # Label
            tk.Label(sign_dialog, text="Enter Sign Name:").pack(pady=10)
            
            # Entry
            sign_entry = tk.Entry(sign_dialog)
            sign_entry.pack(pady=10)
            
            # Variable to track dialog result
            capture_complete = tk.BooleanVar(value=False)
            
            def save_data():
                sign_name = sign_entry.get().strip()
                if not sign_name:
                    messagebox.showwarning("Warning", "Please enter a sign name")
                    return
                
                # Create sign-specific directory
                sign_dir = os.path.join(self.dataset_path, sign_name)
                os.makedirs(sign_dir, exist_ok=True)
                
                # Generate unique filename
                timestamp = int(time.time())
                filename = os.path.join(sign_dir, f"{sign_name}_{timestamp}.json")
                
                # Save landmarks
                with open(filename, 'w') as f:
                    json.dump(landmarks, f)
                
                messagebox.showinfo("Success", f"Captured {sign_name} landmarks")
                capture_complete.set(True)
                sign_dialog.destroy()
            
            # Save button
            tk.Button(sign_dialog, text="Save", command=save_data).pack(pady=10)
            
            # Wait for dialog
            sign_dialog.wait_window()
            
        except Exception as e:
            messagebox.showerror("Capture Error", str(e))




    def start_recognition(self):
        """Start sign language recognition."""
        if self.model is None:
            if not self.load_saved_model():
                return
        
        if self.cap is None or not self.cap.isOpened():
            if not self.init_webcam():
                return
        
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.update_status("Recognition started")
        
        # Start the recognition loop
        self.recognize_sign_language()
    
    def stop_recognition(self):
        """Stop sign language recognition."""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.update_status("Recognition stopped")
    
    def recognize_sign_language(self):
        """
        Perform real-time sign language recognition using MediaPipe and LSTM model.
        Continuously processes webcam frames, detects hand landmarks, and predicts signs.
        """
        try:
            while self.is_running:
                # Read frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    # Use root.after to safely update status from main thread
                    self.root.after(0, lambda: self.update_status("Failed to capture frame"))
                    break

                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)  # Horizontal flip

                # Convert the BGR image to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame with MediaPipe Hands
                results = self.hands.process(rgb_frame)
                
                # Reset hand count
                detected_hands = 0
                current_landmarks = []

                # Draw hands and collect landmarks
                if results.multi_hand_landmarks:
                    detected_hands = len(results.multi_hand_landmarks)
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Collect landmarks for this hand
                        hand_landmark_data = []
                        
                        # Extract x, y, z coordinates for each landmark
                        for landmark in hand_landmarks.landmark:
                            hand_landmark_data.append([
                                landmark.x, 
                                landmark.y, 
                                landmark.z
                            ])
                        
                        current_landmarks.append(hand_landmark_data)
                        print(f"Hands detected in frame: {detected_hands}")
                        
                        # Optionally draw landmarks if checkbox is checked
                        if self.show_landmarks_var.get():
                            self.mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )
                
                # Safely update hands detected label
                self.root.after(0, lambda h=detected_hands: 
                    self.hands_label.config(text=f"Hands Detected: {h}")
                )
                
                # Capture mode for collecting new training data
                if self.capture_var.get() and current_landmarks:
                    self.capture_hand_data(current_landmarks)
                    continue
                
                # Process landmarks for prediction
                if current_landmarks and self.model is not None:
                    # Add landmarks to buffer
                    self.hand_landmarks_buffer.append(current_landmarks)
                    
                    # Only process when buffer is full
                    if len(self.hand_landmarks_buffer) >= self.MAX_BUFFER_SIZE:
                        # Preprocess landmarks
                        processed_landmarks = preprocess_image(self.hand_landmarks_buffer)
                        
                        if processed_landmarks is not None:
                            print("Processed landmarks shape before prediction:", processed_landmarks.shape)
                            if processed_landmarks.shape[1:] != (15, 42):  # Check timesteps and features
                                print(f"ERROR: Shape mismatch. Expected (1, 15, 42), got {processed_landmarks.shape}")
                                self.hand_landmarks_buffer.clear()
                                continue
                            predictions = self.model.predict(processed_landmarks)
                            
                            # Get top prediction
                            top_prediction_index = np.argmax(predictions[0])
                            confidence = predictions[0][top_prediction_index] * 100
                            
                            # Get label from mapping
                            predicted_sign = self.label_map.get(top_prediction_index, "Unknown")
                            
                            # Apply confidence threshold
                            if confidence >= self.confidence_threshold.get():
                                # Update prediction with cooldown
                                self.prediction_cooldown += 1
                                
                                if (self.last_prediction != predicted_sign or 
                                    self.prediction_cooldown >= 5):
                                    # Add to sentence or update prediction
                                    if predicted_sign != " ":
                                        self.sentence.append(predicted_sign)
                                        self.update_sentence_display()
                                    
                                    # Reset tracking
                                    self.last_prediction = predicted_sign
                                    self.prediction_cooldown = 0
                                
                                # Safely update UI predictions
                                self.root.after(0, lambda sign=predicted_sign, conf=confidence: (
                                    self.pred_label.config(text=sign),
                                    self.confidence_label.config(text=f"Confidence: {conf:.2f}%")
                                ))
                            
                        # Clear buffer
                        self.hand_landmarks_buffer.clear()
                
                # Convert frame for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Resize to fit UI
                pil_img = pil_img.resize((640, 480), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(image=pil_img)
                
                # Safely update video label
                self.root.after(0, lambda img=img_tk: (
                    self.video_label.config(image=img),
                    setattr(self.video_label, 'image', img)
                ))
                
                # Process GUI events
                self.root.update()
        
        except Exception as e:
            print(f"Error in recognition: {e}")
            import traceback
            traceback.print_exc()
            
            # Safely show error and stop recognition
            self.root.after(0, lambda exc=e: messagebox.showerror("Recognition Error", str(exc)))
            self.root.after(0, self.stop_recognition)

def main():
    # Set higher DPI awareness for better display on Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
        
    root = tk.Tk()
    app = SignLanguageTranslator(root)
    root.mainloop()

if __name__ == "__main__":
    main()