import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, TimeDistributed, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import threading
import json
import mediapipe as mp
import glob

class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator - Video Training")
        self.root.geometry("800x700")
        
        # Create UI elements
        self.create_widgets()
        
        # Initialize model variables
        self.model = None
        self.history = None
        self.label_map = None
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Dataset configuration
        dataset_frame = ttk.LabelFrame(main_frame, text="Dataset Configuration", padding="10")
        dataset_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(dataset_frame, text="Dataset Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.dataset_path = tk.StringVar(value="dataset")
        path_entry = ttk.Entry(dataset_frame, textvariable=self.dataset_path, width=40)
        path_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(dataset_frame, text="Browse", command=self.browse_dataset).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(dataset_frame, text="Test Split:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.test_split = tk.DoubleVar(value=0.2)
        ttk.Spinbox(dataset_frame, from_=0.1, to=0.5, increment=0.05, textvariable=self.test_split, width=5).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Preprocessing configuration
        preproc_frame = ttk.LabelFrame(main_frame, text="Video Preprocessing", padding="10")
        preproc_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(preproc_frame, text="Frames per Video:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.frames_per_video = tk.IntVar(value=15)
        ttk.Spinbox(preproc_frame, from_=5, to=30, increment=5, textvariable=self.frames_per_video, width=5).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Feature extraction method
        ttk.Label(preproc_frame, text="Feature Extraction:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.feature_extraction = tk.StringVar(value="landmarks")
        ttk.Radiobutton(preproc_frame, text="Hand Landmarks", variable=self.feature_extraction, value="landmarks").grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Model configuration
        model_frame = ttk.LabelFrame(main_frame, text="Model Configuration", padding="10")
        model_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(model_frame, text="LSTM Units:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.lstm_units = tk.IntVar(value=128)
        ttk.Spinbox(model_frame, from_=32, to=256, increment=32, textvariable=self.lstm_units, width=5).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Dropout Rate:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.dropout_rate = tk.DoubleVar(value=0.3)
        ttk.Spinbox(model_frame, from_=0.1, to=0.5, increment=0.1, textvariable=self.dropout_rate, width=5).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Epochs:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.epochs = tk.IntVar(value=50)
        ttk.Spinbox(model_frame, from_=10, to=200, increment=10, textvariable=self.epochs, width=5).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Batch Size:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.batch_size = tk.IntVar(value=16)
        ttk.Spinbox(model_frame, from_=4, to=64, increment=4, textvariable=self.batch_size, width=5).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Progress area
        progress_frame = ttk.LabelFrame(main_frame, text="Training Progress", padding="10")
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = tk.Text(progress_frame, height=15, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar to log text
        scrollbar = ttk.Scrollbar(self.log_text, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=10)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.train_btn = ttk.Button(button_frame, text="Start Training", command=self.start_training_thread)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(button_frame, text="Save Model", command=self.save_model, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Plot Results", command=self.plot_results, state=tk.DISABLED).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Exit", command=self.root.destroy).pack(side=tk.RIGHT, padx=5)

    def browse_dataset(self):
        """Browse for dataset directory."""
        directory = filedialog.askdirectory(initialdir="./", title="Select Dataset Directory")
        if directory:
            self.dataset_path.set(directory)
    
    def log(self, message):
        """Add message to log text widget."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def start_training_thread(self):
        """Start training in a separate thread to keep UI responsive."""
        self.train_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        
        # Start training in a separate thread
        thread = threading.Thread(target=self.train_model)
        thread.daemon = True
        thread.start()
    
    def load_video_dataset(self, dataset_path):
        """Load video dataset and extract features."""
        self.log("Loading video dataset...")
        
        # Initialize MediaPipe hands
        hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        X = []  # Features
        y = []  # Labels
        label_map = {}  # Mapping of class indices to class names
        
        # Get all subfolders in dataset path
        class_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
        
        if not class_folders:
            self.log("No class folders found in dataset path!")
            return [], [], {}
        
        # Sort classes for consistent label assignment
        class_folders.sort()
        
        # Create label map
        for i, folder in enumerate(class_folders):
            label_map[i] = folder
        
        # Number of frames to extract from each video
        frames_per_video = self.frames_per_video.get()
        
        # Process each class folder
        for class_idx, class_folder in enumerate(class_folders):
            self.log(f"Processing class: {class_folder}...")
            
            # Get all video files
            video_path = os.path.join(dataset_path, class_folder)
            video_files = []
            
            # Support multiple video formats
            for ext in ['*.mp4', '*.avi', '*.mov']:
                video_files.extend(glob.glob(os.path.join(video_path, ext)))
            
            if not video_files:
                self.log(f"No video files found in {class_folder}!")
                continue
            
            self.log(f"Found {len(video_files)} videos in {class_folder}")
            
            # Process each video
            for video_file in video_files:
                try:
                    # Open video
                    cap = cv2.VideoCapture(video_file)
                    if not cap.isOpened():
                        self.log(f"Error opening {video_file}")
                        continue
                    
                    # Get video properties
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    if total_frames == 0:
                        self.log(f"No frames in {video_file}")
                        cap.release()
                        continue
                    
                    # Calculate frame indices to extract (evenly distributed)
                    frame_indices = np.linspace(0, total_frames-1, frames_per_video, dtype=int)
                    
                    # Extract landmarks from selected frames
                    landmarks_sequence = []
                    
                    for frame_idx in frame_indices:
                        # Set frame position
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        
                        if not ret:
                            # If frame extraction fails, use zeros
                            landmarks_sequence.append(np.zeros((21*2,)))  # 21 landmarks, x and y
                            continue
                        
                        # Convert to RGB for MediaPipe
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = hands.process(rgb_frame)
                        
                        # Extract landmarks
                        if results.multi_hand_landmarks:
                            # Get first hand landmarks
                            hand = results.multi_hand_landmarks[0]
                            landmarks = []
                            
                            # Extract x, y coordinates (normalized)
                            for landmark in hand.landmark:
                                landmarks.extend([landmark.x, landmark.y])
                            
                            landmarks_sequence.append(np.array(landmarks))
                        else:
                            # No hand detected - use zeros
                            landmarks_sequence.append(np.zeros((21*2,)))
                    
                    # Close video
                    cap.release()
                    
                    # Add to dataset
                    X.append(landmarks_sequence)
                    y.append(class_idx)
                    
                except Exception as e:
                    self.log(f"Error processing {video_file}: {str(e)}")
            
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Release MediaPipe resources
        hands.close()
        
        self.log(f"Dataset shape: X={X.shape}, y={y.shape}")
        return X, y, label_map
    
    def train_model(self):
        """Train the LSTM model for sign language recognition from videos."""
        try:
            # Load dataset
            dataset_path = self.dataset_path.get()
            self.log(f"Loading dataset from {dataset_path}...")
            
            if not os.path.exists(dataset_path):
                self.log(f"Error: Dataset path {dataset_path} does not exist!")
                self.train_btn.config(state=tk.NORMAL)
                return
            
            X, y, self.label_map = self.load_video_dataset(dataset_path)
            
            if len(X) == 0:
                self.log("Error: No valid videos found in the dataset!")
                self.train_btn.config(state=tk.NORMAL)
                return
            
            self.log(f"Loaded {len(X)} videos with {len(self.label_map)} classes:")
            for idx, word in self.label_map.items():
                count = np.sum(y == idx)
                self.log(f"  - {word}: {count} samples")
            
            # One-hot encode target values
            num_classes = len(self.label_map)
            y_one_hot = to_categorical(y, num_classes)
            
            # Split dataset
            test_split = self.test_split.get()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_one_hot, test_size=test_split, random_state=42, stratify=y)
            
            self.log(f"Training set: {len(X_train)} samples")
            self.log(f"Test set: {len(X_test)} samples")
            
            # Build LSTM model
            self.log("Building LSTM model for video sequence data...")
            self.model = Sequential()
            
            # LSTM parameters
            lstm_units = self.lstm_units.get()
            dropout_rate = self.dropout_rate.get()
            
            # Input shape: [frames, features]
            input_shape = (X.shape[1], X.shape[2])
            
            # First LSTM layer
            self.model.add(LSTM(lstm_units, input_shape=input_shape, return_sequences=True))
            self.model.add(Dropout(dropout_rate))
            
            # Second LSTM layer
            self.model.add(LSTM(lstm_units // 2, return_sequences=False))
            self.model.add(Dropout(dropout_rate))
            
            # Dense layers
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(dropout_rate))
            self.model.add(Dense(num_classes, activation='softmax'))
            
            # Compile model
            self.model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
            
            # Log model summary
            model_summary = self.get_model_summary()
            self.log("Model Summary:")
            self.log(model_summary)
            
            # Prepare callbacks
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True),
                # Custom callback for progress updates
                tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: self.update_progress(epoch, self.epochs.get(), logs)
                )
            ]
            
            # Create models directory if it doesn't exist
            if not os.path.exists('models'):
                os.makedirs('models')
            
            # Train model
            self.log("Starting training...")
            epochs = self.epochs.get()
            batch_size = self.batch_size.get()
            
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
            
            # Evaluate model
            self.log("Evaluating model on test data...")
            test_loss, test_acc = self.model.evaluate(X_test, y_test)
            self.log(f"Test accuracy: {test_acc*100:.2f}%")
            self.log(f"Test loss: {test_loss:.4f}")
            
            # Enable save button
            self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
            self.log("Training completed. You can now save the model.")
            
        except Exception as e:
            self.log(f"Error during training: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
        
        finally:
            # Re-enable train button
            self.root.after(0, lambda: self.train_btn.config(state=tk.NORMAL))
    
    def get_model_summary(self):
        """Get model summary as string."""
        # Redirect summary to string
        import io
        summary_io = io.StringIO()
        self.model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
        return summary_io.getvalue()
    
    def update_progress(self, epoch, total_epochs, logs):
        """Update progress bar and log training metrics."""
        # Update progress bar
        progress_pct = (epoch + 1) / total_epochs * 100
        self.root.after(0, lambda: self.progress_var.set(progress_pct))
        
        # Log metrics
        log_msg = f"Epoch {epoch+1}/{total_epochs} - "
        log_msg += f"Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}, "
        log_msg += f"Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}"
        
        self.root.after(0, lambda: self.log(log_msg))
    
    def save_model(self):
        """Save the trained model."""
        if self.model is None:
            messagebox.showerror("Error", "No trained model to save!")
            return
        
        try:
            # Save model
            model_path = os.path.join('models', 'lstm_sign_video_model.h5')
            self.model.save(model_path)
            
            # Save label mapping
            label_map_path = os.path.join('models', 'video_label_map.json')
            with open(label_map_path, 'w') as f:
                # Convert integer keys to strings for JSON
                str_label_map = {str(k): v for k, v in self.label_map.items()}
                json.dump(str_label_map, f)
            
            # Save preprocessing settings
            config_path = os.path.join('models', 'video_config.json')
            config = {
                'frames_per_video': self.frames_per_video.get(),
                'feature_extraction': self.feature_extraction.get()
            }
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            self.log(f"Model saved to {model_path}")
            self.log(f"Label mapping saved to {label_map_path}")
            self.log(f"Configuration saved to {config_path}")
            
            messagebox.showinfo("Success", "Model saved successfully!")
            
        except Exception as e:
            self.log(f"Error saving model: {str(e)}")
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def plot_results(self):
        """Plot training history."""
        if self.history is None:
            messagebox.showerror("Error", "No training history available!")
            return
        
        try:
            # Create figure with subplots
            plt.figure(figsize=(12, 5))
            
            # Plot training & validation accuracy
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'])
            plt.plot(self.history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='lower right')
            
            # Plot training & validation loss
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot results: {str(e)}")

def main():
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()