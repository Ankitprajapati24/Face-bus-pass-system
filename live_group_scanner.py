"""
Live Group Scanner - OPTIMIZED FOR SPEED
Opens camera, detects all faces, shows real-time status, saves marked images
MUCH FASTER face recognition!
"""

import cv2
import json
import os
from datetime import datetime
from pathlib import Path
from face_recognition import FaceRecognitionSystem
import numpy as np

class LiveGroupScanner:
    def __init__(self, database_file='students_database.json', min_confidence=70.0):
        """Initialize live group scanner"""
        self.fr_system = FaceRecognitionSystem(threshold=0.5)  # Slightly relaxed for speed
        self.database_file = database_file
        self.min_confidence = min_confidence
        self.cap = None
        
        # Session cache - remember recognized faces during this session
        self.session_cache = {}
        self.cache_timeout = 10  # Increased from 5 to 10 seconds - remember faces longer
        
        # Create output directory
        Path('data/group_scans').mkdir(parents=True, exist_ok=True)
        
        # Load database
        self.load_database()
        
        # Colors (BGR format)
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)
        self.YELLOW = (0, 255, 255)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        
        print("✅ Live Group Scanner Initialized!")
        print(f"📋 Loaded {len(self.students)} registered students")
        print(f"🎯 Minimum confidence: {self.min_confidence}%")
        print(f"⚡ Speed optimization: ENABLED")
    
    def load_database(self):
        """Load student database"""
        if os.path.exists(self.database_file):
            with open(self.database_file, 'r') as f:
                self.db = json.load(f)
                self.students = self.db.get('students', {})
        else:
            print("⚠️  No database found!")
            self.students = {}
    
    def check_fee_status(self, student_id):
        """Check if student has paid fee"""
        if student_id in self.students:
            return self.students[student_id].get('fee_status') == 'paid'
        return False
    
    def compute_face_hash(self, face_region):
        """
        Create a simple hash of face region for caching
        This helps recognize same person without full AI recognition every frame
        """
        # Resize to small size for fast comparison
        small = cv2.resize(face_region, (50, 50))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Simple hash based on average pixel values
        return hash(tuple(gray.flatten()[::10]))  # Sample every 10th pixel
    
    def get_cached_result(self, face_hash):
        """Check if we've recently recognized this face"""
        if face_hash in self.session_cache:
            cached = self.session_cache[face_hash]
            # Check if cache is still valid (within timeout)
            time_diff = (datetime.now() - cached['timestamp']).total_seconds()
            if time_diff < self.cache_timeout:
                return cached['result']
        return None
    
    def recognize_all_faces(self, frame):
        """
        Recognize all faces in the frame - OPTIMIZED VERSION
        Returns list of results for each face
        """
        # Detect all faces
        faces = self.fr_system.detect_faces(frame)
        
        if len(faces) == 0:
            return []
        
        print(f"⚡ Processing {len(faces)} face(s)...")
        results = []
        
        # Process each face
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face_region = frame[y:y+h, x:x+w]
            
            # Calculate face hash for caching
            face_hash = self.compute_face_hash(face_region)
            
            # Check cache first - SPEED BOOST!
            cached_result = self.get_cached_result(face_hash)
            if cached_result:
                print(f"  Face {i+1}: ⚡ Using cached result")
                # Update coordinates but keep cached recognition
                cached_result['coords'] = (x, y, w, h)
                results.append(cached_result)
                continue
            
            # Not in cache - do full recognition
            print(f"  Face {i+1}: 🔍 Recognizing...")
            
            # OPTIMIZATION: Resize face for faster recognition
            # Smaller image = faster processing
            target_size = 200  # Reduce from original size
            if w > target_size or h > target_size:
                scale = target_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                face_region_resized = cv2.resize(face_region, (new_w, new_h))
            else:
                face_region_resized = face_region
            
            # Save temporary face
            temp_path = f'data/temp_face_{i}.jpg'
            cv2.imwrite(temp_path, face_region_resized)
            
            # Recognize face
            recognition_result = self.fr_system.recognize_face(temp_path)
            
            # Determine status
            if recognition_result['status'] == 'recognized':
                student_id = recognition_result['student_id']
                name = recognition_result['name']
                confidence = recognition_result['confidence']
                
                # Check confidence threshold
                if confidence >= self.min_confidence:
                    # Check fee status
                    fee_paid = self.check_fee_status(student_id)
                    
                    if fee_paid:
                        # PAID
                        status = 'paid'
                        color = self.GREEN
                        label = 'PAID'
                        print(f"    ✅ {name} - PAID ({confidence:.1f}%)")
                    else:
                        # UNPAID
                        status = 'unpaid'
                        color = self.RED
                        label = 'UNPAID'
                        print(f"    ❌ {name} - UNPAID ({confidence:.1f}%)")
                    
                    result = {
                        'coords': (x, y, w, h),
                        'student_id': student_id,
                        'name': name,
                        'confidence': confidence,
                        'status': status,
                        'color': color,
                        'label': label
                    }
                else:
                    # LOW CONFIDENCE - treat as unknown
                    print(f"    ⚠️  Low confidence ({confidence:.1f}% < {self.min_confidence}%)")
                    result = {
                        'coords': (x, y, w, h),
                        'student_id': 'UNKNOWN',
                        'name': 'Unknown',
                        'confidence': confidence,
                        'status': 'unpaid',
                        'color': self.RED,
                        'label': 'UNPAID'
                    }
            else:
                # NOT RECOGNIZED
                print(f"    ❌ Not recognized")
                result = {
                    'coords': (x, y, w, h),
                    'student_id': 'UNKNOWN',
                    'name': 'Unknown',
                    'confidence': 0,
                    'status': 'unpaid',
                    'color': self.RED,
                    'label': 'UNPAID'
                }
            
            # Add to cache
            self.session_cache[face_hash] = {
                'result': result,
                'timestamp': datetime.now()
            }
            
            results.append(result)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return results
    
    def draw_face_box(self, frame, result):
        """Draw bounding box and label for one face"""
        x, y, w, h = result['coords']
        color = result['color']
        label = result['label']
        name = result['name']
        
        # Draw thick rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 4)
        
        # Prepare texts
        if name != 'Unknown':
            top_text = label
            bottom_text = name
        else:
            top_text = label
            bottom_text = "Unknown Person"
        
        # Calculate sizes
        font = cv2.FONT_HERSHEY_SIMPLEX
        (top_w, top_h), _ = cv2.getTextSize(top_text, font, 0.8, 2)
        (bot_w, bot_h), _ = cv2.getTextSize(bottom_text, font, 0.6, 2)
        
        max_width = max(top_w, bot_w) + 20
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (x, y - 60), 
                     (x + max_width, y - 5),
                     color, -1)
        
        # Draw white border for better visibility
        cv2.rectangle(frame, 
                     (x, y - 60), 
                     (x + max_width, y - 5),
                     self.WHITE, 2)
        
        # Draw texts
        cv2.putText(frame, top_text, 
                   (x + 10, y - 35), 
                   font, 0.8, self.WHITE, 2)
        
        cv2.putText(frame, bottom_text, 
                   (x + 10, y - 12), 
                   font, 0.6, self.WHITE, 2)
        
        return frame
    
    def draw_info_panel(self, frame, face_count, scan_mode=False):
        """Draw information panel at top of frame"""
        height, width = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), self.BLACK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title - Fixed display
        title = "LIVE GROUP SCANNER" if not scan_mode else "SCANNING..."
        cv2.putText(frame, title, 
                   (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.WHITE, 2)
        
        # Face count
        cv2.putText(frame, f"Faces: {face_count}", 
                   (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.YELLOW, 2)
        
        # Instructions
        if not scan_mode:
            cv2.putText(frame, "SPACE: SCAN | 'q': QUIT", 
                       (width - 380, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.WHITE, 2)
        else:
            cv2.putText(frame, "Processing...", 
                       (width - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.YELLOW, 2)
        
        return frame
    
    def save_marked_image(self, frame, results):
        """Save the marked image with all annotations"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"group_scan_{timestamp}.jpg"
        filepath = f"data/group_scans/{filename}"
        
        cv2.imwrite(filepath, frame)
        
        # Also save a text report
        report_filename = f"group_scan_{timestamp}.txt"
        report_filepath = f"data/group_scans/{report_filename}"
        
        with open(report_filepath, 'w', encoding='utf-8') as f:  # Added UTF-8 encoding
            f.write("=" * 60 + "\n")
            f.write("GROUP SCAN REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Faces: {len(results)}\n")
            f.write(f"Paid: {sum(1 for r in results if r['status'] == 'paid')}\n")
            f.write(f"Unpaid/Unknown: {sum(1 for r in results if r['status'] == 'unpaid')}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("INDIVIDUAL RESULTS:\n")
            f.write("=" * 60 + "\n\n")
            
            for i, result in enumerate(results, 1):
                status_text = "[PAID]" if result['status'] == 'paid' else "[UNPAID]"  # Changed from emojis
                f.write(f"{i}. {status_text}\n")
                f.write(f"   Name: {result['name']}\n")
                if result['name'] != 'Unknown':
                    f.write(f"   Student ID: {result['student_id']}\n")
                    f.write(f"   Confidence: {result['confidence']:.1f}%\n")
                f.write("\n")
        
        return filepath, report_filepath
    
    def start_scanner(self):
        """Start the live group scanner"""
        print("\n" + "=" * 60)
        print("🚌 LIVE GROUP SCANNER - SPEED OPTIMIZED ⚡")
        print("=" * 60)
        print("\n📹 Camera will open and show all detected faces")
        print("📸 Press SPACE to scan all faces and save marked image")
        print("❌ Press 'q' to quit")
        print("\n⚡ OPTIMIZATIONS ENABLED:")
        print("   • Session caching (faster repeat scans)")
        print("   • Reduced image size for recognition")
        print("   • Quick face detection")
        print("\n" + "=" * 60 + "\n")
        
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ Could not open webcam!")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Reduce FPS for better performance
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera started successfully!")
        print("\n📍 Position all people in frame")
        print("📸 Press SPACE when ready to scan\n")
        
        # Variables for smooth display
        last_faces = []  # Remember last detected faces to avoid flicker
        detection_interval = 10  # Detect faces every 10 frames (was 5 - more stable now)
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Detect faces periodically (not every frame) to reduce flicker
            if frame_count % detection_interval == 0:
                last_faces = self.fr_system.detect_faces(frame)
            
            # Create display frame
            display_frame = frame.copy()
            
            # Always draw the last detected faces (smooth display)
            for (x, y, w, h) in last_faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), self.YELLOW, 2)
                cv2.putText(display_frame, "Face", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.YELLOW, 2)
            
            # Draw info panel (always, for smooth display)
            display_frame = self.draw_info_panel(display_frame, len(last_faces), scan_mode=False)
            
            # Show frame with smooth refresh
            cv2.imshow('Live Group Scanner', display_frame)
            
            # Handle key press (15ms for even smoother display)
            key = cv2.waitKey(15) & 0xFF
            
            if key == 32:  # SPACE - Scan all faces
                # Get fresh detection
                current_faces = self.fr_system.detect_faces(frame)
                
                if len(current_faces) == 0:
                    print("⚠️  No faces detected! Position people in frame.")
                    continue
                
                print("\n" + "=" * 60)
                print(f"⚡ FAST SCANNING {len(current_faces)} FACE(S)...")
                print("=" * 60)
                
                # Show scanning message
                scan_frame = frame.copy()
                scan_frame = self.draw_info_panel(scan_frame, len(current_faces), scan_mode=True)
                cv2.imshow('Live Group Scanner', scan_frame)
                cv2.waitKey(100)
                
                # Recognize all faces - NOW MUCH FASTER!
                start_time = datetime.now()
                results = self.recognize_all_faces(frame)
                end_time = datetime.now()
                
                processing_time = (end_time - start_time).total_seconds()
                print(f"⚡ Processing completed in {processing_time:.2f} seconds")
                
                # Draw all boxes on the frame
                marked_frame = frame.copy()
                for result in results:
                    marked_frame = self.draw_face_box(marked_frame, result)
                
                # Draw info panel on marked frame
                marked_frame = self.draw_info_panel(marked_frame, len(results), scan_mode=False)
                
                # Save the marked image
                image_path, report_path = self.save_marked_image(marked_frame, results)
                
                # Print results
                print("\n" + "=" * 60)
                print("📊 SCAN RESULTS")
                print("=" * 60)
                print(f"Total Faces: {len(results)}")
                print(f"✅ Paid: {sum(1 for r in results if r['status'] == 'paid')}")
                print(f"❌ Unpaid/Unknown: {sum(1 for r in results if r['status'] == 'unpaid')}")
                
                print("\nIndividual Results:")
                for i, result in enumerate(results, 1):
                    status_icon = "✅" if result['status'] == 'paid' else "❌"
                    print(f"{i}. {status_icon} {result['label']}: {result['name']}")
                    if result['name'] != 'Unknown':
                        print(f"   Confidence: {result['confidence']:.1f}%")
                
                print("\n" + "=" * 60)
                print(f"💾 Marked image saved: {image_path}")
                print(f"📄 Report saved: {report_path}")
                print("=" * 60)
                
                # Display marked image for 3 seconds
                print("\n📺 Showing marked image... (3 seconds)")
                cv2.imshow('Live Group Scanner', marked_frame)
                cv2.waitKey(3000)
                
                print("\n✅ Ready for next scan! (Cached results = faster!)")
            
            elif key == ord('q'):  # Quit
                print("\n👋 Exiting scanner...")
                break
            
            elif key == ord('c'):  # Clear cache
                self.session_cache.clear()
                print("🗑️  Session cache cleared!")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n✅ Scanner closed!")


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("👥 LIVE GROUP SCANNER - SPEED OPTIMIZED ⚡")
    print("=" * 60)
    
    # Adjust confidence threshold - lower = more accepting
    # 65% works well for real-world conditions (glasses, angles, lighting)
    MIN_CONFIDENCE = 65.0  # Changed from 70.0 to 65.0
    
    # Initialize scanner
    scanner = LiveGroupScanner(min_confidence=MIN_CONFIDENCE)
    
    print("\n📋 Registered Students:")
    for student_id, data in scanner.students.items():
        status = "✅ PAID" if data.get('fee_status') == 'paid' else "❌ UNPAID"
        print(f"  {student_id}: {data['name']} - {status}")
    
    print("\n" + "=" * 60)
    print("🎥 Starting camera in 3 seconds...")
    print("=" * 60)
    
    import time
    time.sleep(3)
    
    # Start the scanner
    scanner.start_scanner()


if __name__ == "__main__":
    main()