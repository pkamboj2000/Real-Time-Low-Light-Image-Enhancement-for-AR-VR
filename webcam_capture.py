import cv2
import numpy as np
import os
from pathlib import Path
import time
import json
from datetime import datetime

class WebcamDataCapture:
    """Capture low-light video data from webcam for training/testing"""
    
    def __init__(self, output_dir="webcam_data", frame_size=(640, 480)):
        self.output_dir = Path(output_dir)
        self.frame_size = frame_size
        self.setup_directories()
        
    def setup_directories(self):
        """Create output directories"""
        (self.output_dir / "low_light").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "normal_light").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metadata").mkdir(parents=True, exist_ok=True)
        
    def capture_session(self, duration_seconds=60, fps=30):
        """Capture a session of low-light and normal light frames"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_metadata = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "duration_seconds": duration_seconds,
            "fps": fps,
            "frame_size": self.frame_size,
            "frames": []
        }
        
        print(f"Starting capture session: {session_id}")
        print("Press 'q' to quit, 's' to save frame pair, 'l' for low-light mode")
        
        frame_count = 0
        low_light_mode = False
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Display current frame
            display_frame = frame.copy()
            
            # Add overlay text
            status = "LOW-LIGHT MODE" if low_light_mode else "NORMAL MODE"
            color = (0, 0, 255) if low_light_mode else (0, 255, 0)
            cv2.putText(display_frame, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 's' to save, 'l' to toggle mode, 'q' to quit", 
                       (10, display_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Webcam Capture', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('l'):
                low_light_mode = not low_light_mode
                print(f"Switched to {'LOW-LIGHT' if low_light_mode else 'NORMAL'} mode")
            elif key == ord('s'):
                # Save frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                
                if low_light_mode:
                    # Simulate low-light by reducing brightness and adding noise
                    low_frame = self.simulate_low_light(frame)
                    filename = f"{session_id}_{timestamp}_low.png"
                    filepath = self.output_dir / "low_light" / filename
                    cv2.imwrite(str(filepath), low_frame)
                    
                    # Also save original as reference
                    ref_filename = f"{session_id}_{timestamp}_ref.png"
                    ref_filepath = self.output_dir / "normal_light" / ref_filename
                    cv2.imwrite(str(ref_filepath), frame)
                    
                    print(f"Saved low-light pair: {filename}")
                    
                    session_metadata["frames"].append({
                        "frame_id": frame_count,
                        "timestamp": timestamp,
                        "type": "low_light_pair",
                        "low_light_file": filename,
                        "reference_file": ref_filename
                    })
                else:
                    filename = f"{session_id}_{timestamp}_normal.png"
                    filepath = self.output_dir / "normal_light" / filename
                    cv2.imwrite(str(filepath), frame)
                    print(f"Saved normal frame: {filename}")
                    
                    session_metadata["frames"].append({
                        "frame_id": frame_count,
                        "timestamp": timestamp,
                        "type": "normal_light",
                        "file": filename
                    })
                
                frame_count += 1
            
            # Auto-quit after duration
            if time.time() - start_time > duration_seconds:
                print(f"Session duration reached ({duration_seconds}s)")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save session metadata
        session_metadata["end_time"] = datetime.now().isoformat()
        session_metadata["total_frames"] = frame_count
        
        metadata_file = self.output_dir / "metadata" / f"{session_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(session_metadata, f, indent=2)
        
        print(f"Session complete! Captured {frame_count} frame pairs")
        print(f"Metadata saved to: {metadata_file}")
        
        return session_metadata
    
    def simulate_low_light(self, frame, brightness_factor=0.3, noise_level=15):
        """Simulate low-light conditions"""
        # Reduce brightness
        dark_frame = (frame.astype(np.float32) * brightness_factor).astype(np.uint8)
        
        # Add noise
        noise = np.random.normal(0, noise_level, frame.shape).astype(np.int16)
        noisy_frame = np.clip(dark_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return noisy_frame
    
    def create_video_from_session(self, session_id, output_fps=30):
        """Create comparison video from captured session"""
        metadata_file = self.output_dir / "metadata" / f"{session_id}_metadata.json"
        
        if not metadata_file.exists():
            print(f"Metadata file not found: {metadata_file}")
            return
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        # Get frame pairs
        frame_pairs = [f for f in metadata["frames"] if f["type"] == "low_light_pair"]
        
        if not frame_pairs:
            print("No frame pairs found in session")
            return
        
        # Create video writer
        output_file = self.output_dir / f"{session_id}_comparison.mp4"
        
        # Read first frame to get dimensions
        first_low = cv2.imread(str(self.output_dir / "low_light" / frame_pairs[0]["low_light_file"]))
        first_ref = cv2.imread(str(self.output_dir / "normal_light" / frame_pairs[0]["reference_file"]))
        
        # Create side-by-side frame
        h, w = first_low.shape[:2]
        combined_frame = np.hstack([first_low, first_ref])
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_file), fourcc, output_fps, (combined_frame.shape[1], combined_frame.shape[0]))
        
        for pair in frame_pairs:
            low_frame = cv2.imread(str(self.output_dir / "low_light" / pair["low_light_file"]))
            ref_frame = cv2.imread(str(self.output_dir / "normal_light" / pair["reference_file"]))
            
            if low_frame is not None and ref_frame is not None:
                # Add labels
                cv2.putText(low_frame, "Low Light", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(ref_frame, "Reference", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                combined = np.hstack([low_frame, ref_frame])
                out.write(combined)
        
        out.release()
        print(f"Comparison video saved: {output_file}")

def main():
    """Interactive webcam capture session"""
    print("Webcam Data Capture for AR/VR Low-Light Enhancement")
    print("=" * 60)
    
    capture = WebcamDataCapture()
    
    print("\nInstructions:")
    print("1. Ensure good lighting initially")
    print("2. Press 'l' to toggle low-light simulation mode")
    print("3. Press 's' to save frame pairs")
    print("4. Press 'q' to quit")
    print("\nStarting capture in 3 seconds...")
    
    time.sleep(3)
    
    # Start capture session (5 minutes)
    session_data = capture.capture_session(duration_seconds=300, fps=30)
    
    if session_data and session_data["total_frames"] > 0:
        print("\nCreating comparison video...")
        capture.create_video_from_session(session_data["session_id"])
        
        print(f"\nCapture complete!")
        print(f"Data saved in: {capture.output_dir}")
        print(f"Total frame pairs: {session_data['total_frames']}")

if __name__ == "__main__":
    main()
