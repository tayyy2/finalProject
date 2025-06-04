import cv2
import os
import time
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import mediapipe as mp
import threading

def create_folders(words):
    """Create folders for each word if they don't exist."""
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    for word in words:
        word_folder = os.path.join(dataset_folder, word)
        if not os.path.exists(word_folder):
            os.makedirs(word_folder)

def get_next_video_number(word_folder):
    """Get the next available video number for a specific word."""
    existing_videos = [f for f in os.listdir(word_folder) if f.endswith('.mp4')]
    if not existing_videos:
        return 0
    
    # Extract numbers from filenames like "word_X.mp4"
    existing_numbers = []
    for video in existing_videos:
        try:
            number = int(video.split('_')[1].split('.')[0])
            existing_numbers.append(number)
        except (IndexError, ValueError):
            continue
    
    return max(existing_numbers) + 1 if existing_numbers else 0

def collect_data():
    """Main function to collect data for sign language translation."""
    # List of words to collect data for
    words = ["Again","Bathroom","Eat","Hello","Help","How are you","No","Please","Sorry","thanks","What","yes"]
    create_folders(words)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Get webcam properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:  # Fallback if FPS detection fails
        fps = 30
    
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # GUI for word selection
    root = tk.Tk()
    root.title("ເກັບຂໍ້ມູນພາສາມື")
    root.geometry("500x650")
    
    # Variables to track current word and sample count
    current_word = tk.StringVar(value="")
    sample_count = tk.IntVar(value=0)
    max_samples = tk.IntVar(value=10)
    show_landmarks = tk.BooleanVar(value=True)
    recording = tk.BooleanVar(value=False)
    
    # Video recording settings
    video_duration = tk.IntVar(value=3)  # Duration in seconds
    
    # Create label frame for instructions
    instruction_frame = tk.LabelFrame(root, text="Instructions", padx=20, pady=20)
    instruction_frame.pack(fill=tk.X, padx=20, pady=20)
    
    instructions = tk.Label(instruction_frame, 
                           text="1. ເລືອກຄຳດ້ານລຸ່ມ \n"
                                "2. ວາງມຶໃຫ້ຕົງກັບກ້ອງ \n"
                                "3. ກົດ 'v' ເພື່ອອັດວີດິໂອ \n"
                                "4. ກົດ ESC ເພື່ອອອກ ")
    instructions.pack()

    
    # Create frame for word selection
    word_frame = tk.LabelFrame(root, text=" ເລືອກຄຳທີຈະເກັບ  ", padx=10, pady=10)
    word_frame.pack(fill=tk.X, padx=10, pady=10)

    # Create a grid layout for the word buttons
    row, col = 0, 0
    max_cols = 3  # Number of columns in the grid

    for word in words:
        btn = tk.Button(word_frame, text=word.replace('_', ' ').title(), 
                       command=lambda w=word: select_word(w), width=10)
        btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
        
        # Move to the next column or row
        col += 1
        if col >= max_cols:
            col = 0
            row += 1
    
    # Create frame for status information
    status_frame = tk.LabelFrame(root, text="ສະຖານະ", padx=10, pady=10)
    status_frame.pack(fill=tk.X, padx=10, pady=10)
    
    current_word_label = tk.Label(status_frame, text="ຄຳປັດຈຸບັນ : ບໍ່ມີ")
    current_word_label.pack(fill=tk.X, pady=5)
    
    progress_label = tk.Label(status_frame, text="ກຳລັງເກັບ: 0/0")
    progress_label.pack(fill=tk.X, pady=5)
    
    total_videos_label = tk.Label(status_frame, text="ວິດີໂອທັງໝົດ : 0")
    total_videos_label.pack(fill=tk.X, pady=5)
    
    hands_detected_label = tk.Label(status_frame, text="ກວດຈັບມື: 0")
    hands_detected_label.pack(fill=tk.X, pady=5)
    
    recording_status = tk.Label(status_frame, text="ຍັງບໍ່ໄດ້ເປີດ", fg="black")
    recording_status.pack(fill=tk.X, pady=5)
    
    # Options frame
    options_frame = tk.LabelFrame(root, text="Options", padx=10, pady=10)
    options_frame.pack(fill=tk.X, padx=10, pady=10)
    
    # Show landmarks checkbox
    landmarks_check = tk.Checkbutton(options_frame, text="ສະແດງຂໍ້ຕໍ່ນິ້ວ-ມື", variable=show_landmarks)
    landmarks_check.pack(anchor=tk.W, pady=5)
    
    # Video duration control
    duration_frame = tk.Frame(options_frame)
    duration_frame.pack(fill=tk.X, pady=5)
    
    tk.Label(duration_frame, text="ຄວາມຍາວວິດີໂອ (ວິນາທີ):").pack(side=tk.LEFT)
    duration_entry = tk.Entry(duration_frame, textvariable=video_duration, width=5)
    duration_entry.pack(side=tk.LEFT, padx=5)
    
    # Entry for number of samples
    samples_frame = tk.Frame(options_frame)
    samples_frame.pack(fill=tk.X, pady=5)
    
    tk.Label(samples_frame, text="ຈຳນວນວິດີໂອທີ່ຈະເກັບ:").pack(side=tk.LEFT)
    samples_entry = tk.Entry(samples_frame, textvariable=max_samples, width=5)
    samples_entry.pack(side=tk.LEFT, padx=5)
    
    # Add a preview frame
    preview_frame = tk.LabelFrame(root, text="Preview", padx=10, pady=10)
    preview_frame.pack(fill=tk.X, padx=10, pady=10)
    
    preview_progress = ttk.Progressbar(preview_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
    preview_progress.pack(fill=tk.X, pady=10)
    
    # Function to count existing videos for a word
    def count_existing_videos(word):
        if not word:
            return 0
        word_folder = os.path.join("dataset", word)
        if not os.path.exists(word_folder):
            return 0
        return len([f for f in os.listdir(word_folder) if f.endswith('.mp4')])
    
    # Function to update total videos label
    def update_total_videos_label(word):
        count = count_existing_videos(word)
        total_videos_label.config(text=f"Total Videos: {count}")
        return count
    
    # Function to select a word
    def select_word(word):
        current_word.set(word)
        sample_count.set(0)
        current_word_label.config(text=f"ຄຳປັດຈຸບັນ: {word.replace('_', ' ').title()}")
        # Get existing video count
        existing_count = update_total_videos_label(word)
        # Display information
        messagebox.showinfo("Word Selected", 
                           f"Selected '{word.replace('_', ' ').title()}'\n"
                           f"Found {existing_count} existing videos.\n"
                           f"Will collect {max_samples.get()} more videos.")
        update_progress_label()
    
    # Function to update progress label
    def update_progress_label():
        progress_label.config(text=f"Progress: {sample_count.get()}/{max_samples.get()}")
    
    # Function to update hands detected label
    def update_hands_label(count):
        hands_detected_label.config(text=f"ກວດຈັບມືໄດ້: {count}")
    
    # Function to update recording status
    def update_recording_status(is_recording, progress=0):
        if is_recording:
            recording_status.config(text=f"ກຳລັງບັນທຶກ {progress}%", fg="red")
            preview_progress['value'] = progress
        else:
            recording_status.config(text="ບໍ່ບັນທຶກ", fg="black")
            preview_progress['value'] = 0
    
    # Function to handle window closing
    def on_closing():
        hands.close()  # Release MediaPipe resources
        cap.release()
        cv2.destroyAllWindows()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start camera display in a separate thread
    thread = threading.Thread(target=camera_loop, args=(
        cap, 
        current_word, 
        sample_count, 
        max_samples, 
        update_progress_label, 
        update_hands_label, 
        update_recording_status,
        update_total_videos_label,
        show_landmarks,
        video_duration,
        recording,
        hands,
        mp_hands,
        mp_drawing,
        mp_drawing_styles,
        fps
    ))
    thread.daemon = True
    thread.start()
    
    root.mainloop()

def camera_loop(cap, current_word, sample_count, max_samples, update_progress_callback, 
                update_hands_callback, update_recording_callback, update_total_videos_callback,
                show_landmarks, video_duration, recording, hands, mp_hands, mp_drawing, 
                mp_drawing_styles, fps):
    """Main camera loop with key press detection and hand landmark visualization."""
    
    cv2.namedWindow("ໜ້າຕ່າງກ້ອງ", cv2.WINDOW_NORMAL)
    
    # Video writer variables
    video_writer = None
    frames_to_record = 0
    recorded_frames = 0
    start_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Flip frame horizontally for more intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Draw guide rectangle
        h, w = frame.shape[:2]
        guide_margin = 80
        cv2.rectangle(display_frame, 
                     (guide_margin, guide_margin),
                     (w - guide_margin, h - guide_margin),
                     (0, 255, 0), 2)
        
        # Process with MediaPipe to detect hand landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        # Count detected hands
        num_hands = 0
        
        # Draw hand landmarks if enabled
        if results.multi_hand_landmarks and show_landmarks.get():
            num_hands = len(results.multi_hand_landmarks)
            
            # Update hands detected label
            update_hands_callback(num_hands)
            
            # Draw landmarks for each detected hand
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get hand type (left or right)
                hand_type = handedness.classification[0].label
                
                # Choose color based on hand type (Red for left, Blue for right)
                color = (0, 0, 255) if hand_type == "Left" else (255, 0, 0)
                
                # Draw the landmarks and connections
                mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Add hand type label near the wrist
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_x = int(wrist.x * w)
                wrist_y = int(wrist.y * h)
                cv2.putText(
                    display_frame,
                    hand_type,
                    (wrist_x - 30, wrist_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
        else:
            update_hands_callback(0)
            
        # Display word and count
        word = current_word.get()
        count = sample_count.get()
        max_count = max_samples.get()
        
        if word:
            if recording.get():
                status_text = f"ຄຳ: {word} | ກຳລັງບັນທຶກ: {count+1}/{max_count}"
            else:
                status_text = f"ຄຳ: {word} | Progress: {count}/{max_count} | ກົດ 'v' ເພຶ່ອບັນທຶກ"
        else:
            status_text = "ເລືອກຄຳຈາກລາຍການ"
        
        cv2.putText(display_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Handle video recording
        if recording.get():
            if video_writer is not None:
                # Extract ROI (hand area)
                roi = frame[guide_margin:h-guide_margin, guide_margin:w-guide_margin]
                
                # Write frame to video
                video_writer.write(roi)
                
                # Update recording progress
                recorded_frames += 1
                elapsed_time = time.time() - start_time
                progress = min(int((elapsed_time / video_duration.get()) * 100), 100)
                update_recording_callback(True, progress)
                
                # Draw recording indicator
                cv2.circle(display_frame, (w - 30, 30), 10, (0, 0, 255), -1)
                
                # Check if recording is complete
                if elapsed_time >= video_duration.get():
                    recording.set(False)
                    video_writer.release()
                    video_writer = None
                    
                    # Update count
                    sample_count.set(count + 1)
                    update_progress_callback()
                    update_recording_callback(False)
                    update_total_videos_callback(word)
                    
                    # Display saved message briefly
                    cv2.putText(display_frame, "ບັນທຶກວິດີໂອສຳເລັດ", (w//2 - 100, h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow("ໜ້າຕ່າງກ້ອງ", display_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Process key presses
        if key == 27:  # ESC key - reset current collection
            current_word.set("")
            if recording.get():
                recording.set(False)
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                update_recording_callback(False)
        
        elif key == ord('v') and word and not recording.get():  # 'v' key - start video recording
            if count < max_count:
                # Get word folder
                word_folder = os.path.join("dataset", word)
                
                # Get next available video number
                next_num = get_next_video_number(word_folder)
                
                # Create video path
                video_path = os.path.join(word_folder, f"{word}_{next_num}.mp4")
                
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' if mp4v doesn't work
                roi_h, roi_w = h - 2*guide_margin, w - 2*guide_margin
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (roi_w, roi_h))
                
                recording.set(True)
                start_time = time.time()
                recorded_frames = 0
    
    # Cleanup
    if video_writer is not None:
        video_writer.release()
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

def extract_landmarks_from_video(video_path):
    """Extract and save hand landmarks from a recorded video."""
    # This function can be implemented to process videos 
    # and extract landmark coordinates for training
    pass

if __name__ == "__main__":
    collect_data()