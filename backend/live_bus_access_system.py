"""
Live Bus Access System - FIXED VERSION
Real-time face recognition with proper face matching
"""

import cv2
import json
import os
from datetime import datetime
from pathlib import Path
from face_recognition import FaceRecognitionSystem

class LiveBusAccessSystem:
    def __init__(self, database_file='students_database.json', min_confidence=70.0):
        """
        Initialize the live access system
        
        Args:
            database_file: JSON database file path
            min_confidence: Minimum confidence % to accept recognition (default: 70%)
        """
        self.fr_system = FaceRecognitionSystem(threshold=0.4)  # Lower threshold for DeepFace
        self.database_file = database_file
        self.cap = None
        self.min_confidence = min_confidence  # OUR confidence threshold
        
        # Create directories
        Path('data/unpaid_captures').mkdir(parents=True, exist_ok=True)
        
        # Load database
        self.load_database()
        
        # Colors (BGR format for OpenCV)
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)
        self.YELLOW = (0, 255, 255)
        self.WHITE = (255, 255, 255)
        
        print("🚌 Live Bus Access System Initialized!")
        print(f"📋 Database loaded with {len(self.students)} student records")
        print(f"🎯 Minimum confidence threshold: {self.min_confidence}%")
        
        # Show registered faces (actual face images)
        registered = self.fr_system.get_registered_students()
        print(f"👤 Registered faces: {len(registered)}")
        for student in registered:
            print(f"  - {student['student_id']}: {student['name']}")
    
    def load_database(self):
        """Load student database from JSON file"""
        if os.path.exists(self.database_file):
            with open(self.database_file, 'r') as f:
                self.db = json.load(f)
                self.students = self.db.get('students', {})
        else:
            print("⚠️  Database file not found! Creating new one...")
            self.db = {
                'students': {},
                'access_logs': [],
                'unpaid_captures': []
            }
            self.students = {}
            self.save_database()
    
    def save_database(self):
        """Save database to JSON file"""
        with open(self.database_file, 'w') as f:
            json.dump(self.db, f, indent=2)
    
    def check_fee_status(self, student_id):
        """Check if student has paid fee"""
        if student_id in self.students:
            student = self.students[student_id]
            return student.get('fee_status') == 'paid'
        return False
    
    def log_access(self, student_id, name, status, confidence=0):
        """Log access attempt"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'student_id': student_id,
            'name': name,
            'status': status,
            'confidence': confidence
        }
        self.db['access_logs'].append(log_entry)
        self.save_database()
    
    def save_unpaid_capture(self, image, student_id=None, name="Unknown"):
        """Save image of unpaid/unknown person"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if student_id:
            filename = f"unpaid_{student_id}_{timestamp}.jpg"
        else:
            filename = f"unknown_{timestamp}.jpg"
        
        filepath = f"data/unpaid_captures/{filename}"
        cv2.imwrite(filepath, image)
        
        # Log to database
        capture_entry = {
            'timestamp': datetime.now().isoformat(),
            'student_id': student_id or 'UNKNOWN',
            'name': name,
            'filename': filename,
            'filepath': filepath
        }
        self.db['unpaid_captures'].append(capture_entry)
        self.save_database()
        
        print(f"📸 Saved capture: {filename}")
        return filepath
    
    def draw_status_box(self, frame, x, y, w, h, status, name, confidence=0):
        """Draw colored box and status text on frame"""
        
        if status == "ALLOWED":
            color = self.GREEN
            text = "ACCESS GRANTED"
            detail = f"{name} ({confidence:.1f}%)"
        elif status == "DENIED":
            color = self.RED
            text = "NO PASS - UNPAID"
            detail = f"{name} - PAY FEE!"
        else:  # UNKNOWN
            color = self.RED
            text = "NOT REGISTERED"
            detail = "Unknown Person"
        
        # Draw thick rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Calculate text size for background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX  # Use SIMPLEX instead of BOLD
        (text_width, text_height), _ = cv2.getTextSize(text, font, 0.8, 2)
        (detail_width, detail_height), _ = cv2.getTextSize(detail, font, 0.6, 2)
        
        max_width = max(text_width, detail_width)
        
        # Draw background rectangle for text (solid)
        cv2.rectangle(frame, 
                     (x, y - 50), 
                     (x + max_width + 20, y - 5),
                     color, -1)
        
        # Main status text (white text on colored background)
        cv2.putText(frame, text, 
                   (x + 10, y - 28), 
                   font, 0.7, self.WHITE, 2)
        
        # Detail text
        cv2.putText(frame, detail, 
                   (x + 10, y - 10), 
                   font, 0.5, self.WHITE, 1)
        
        return frame
    
    def start_live_system(self):
        """Start the live bus access system"""
        print("\n" + "=" * 60)
        print("🚌 LIVE BUS ACCESS SYSTEM STARTED")
        print("=" * 60)
        print("\n📹 Camera continuously scanning for faces")
        print("✅ Registered + Paid: GREEN box + Access Granted")
        print("❌ Unknown/Unpaid: RED box + Photo saved automatically")
        print("\nPress 'q' to quit")
        print("=" * 60 + "\n")
        
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ Could not open webcam!")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        last_recognition_time = datetime.now()
        recognition_cooldown = 2  # seconds between recognitions
        last_result = None
        show_result_until = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            display_frame = frame.copy()
            current_time = datetime.now()
            
            # Detect faces in current frame
            faces = self.fr_system.detect_faces(frame)
            
            # If we're showing a result, keep showing it
            if show_result_until and current_time < show_result_until and last_result:
                x, y, w, h = last_result['face_coords']
                display_frame = self.draw_status_box(
                    display_frame, x, y, w, h,
                    last_result['status'],
                    last_result['name'],
                    last_result.get('confidence', 0)
                )
            else:
                # Draw yellow box around detected faces (scanning mode)
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), self.YELLOW, 2)
                    cv2.putText(display_frame, "Scanning...", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.YELLOW, 2)
            
            # Check if we should run recognition
            time_since_last = (current_time - last_recognition_time).total_seconds()
            
            # Auto-recognize if face detected and cooldown passed
            if len(faces) > 0 and time_since_last > recognition_cooldown:
                print(f"\n🔍 Face detected! Running recognition...")
                
                # Save temporary image
                temp_path = 'data/temp_recognition.jpg'
                cv2.imwrite(temp_path, frame)
                
                # USE FACE RECOGNITION SYSTEM TO MATCH WITH REGISTERED FACES
                result = self.fr_system.recognize_face(temp_path)
                
                # Get face coordinates for drawing
                x, y, w, h = faces[0]
                
                if result['status'] == 'recognized':
                    # FACE MATCHED with a registered face image!
                    student_id = result['student_id']
                    name = result['name']
                    confidence = result['confidence']
                    
                    print(f"🔍 Face matched: {name} (ID: {student_id}) - {confidence:.1f}%")
                    
                    # CHECK CONFIDENCE THRESHOLD
                    if confidence < self.min_confidence:
                        # TOO LOW CONFIDENCE - Treat as UNKNOWN
                        print(f"⚠️  Confidence too low ({confidence:.1f}% < {self.min_confidence}%) - Treating as UNKNOWN")
                        self.log_access("UNKNOWN", "Unknown", "unrecognized_low_confidence", confidence)
                        
                        # Save their photo automatically
                        self.save_unpaid_capture(frame, None, f"Low_Conf_{name}")
                        
                        last_result = {
                            'status': 'UNKNOWN',
                            'name': 'Unknown',
                            'confidence': 0,
                            'face_coords': (x, y, w, h)
                        }
                    else:
                        # CONFIDENCE HIGH ENOUGH - Check fee status
                        fee_paid = self.check_fee_status(student_id)
                        
                        if fee_paid:
                            # ALLOWED - Registered AND Paid
                            print(f"✅ ACCESS GRANTED: {name}")
                            self.log_access(student_id, name, "allowed", confidence)
                            
                            last_result = {
                                'status': 'ALLOWED',
                                'name': name,
                                'confidence': confidence,
                                'face_coords': (x, y, w, h)
                            }
                            
                        else:
                            # DENIED - Registered but NOT Paid
                            print(f"❌ ACCESS DENIED: {name} - FEE NOT PAID!")
                            self.log_access(student_id, name, "denied_unpaid", confidence)
                            
                            # Save their photo
                            self.save_unpaid_capture(frame, student_id, name)
                            
                            last_result = {
                                'status': 'DENIED',
                                'name': name,
                                'confidence': confidence,
                                'face_coords': (x, y, w, h)
                            }
                
                else:
                    # UNKNOWN - Face did NOT match any registered face
                    print(f"❌ UNKNOWN PERSON - Not in registered faces!")
                    self.log_access("UNKNOWN", "Unknown", "unrecognized", 0)
                    
                    # Save their photo automatically
                    self.save_unpaid_capture(frame, None, "Unknown")
                    
                    last_result = {
                        'status': 'UNKNOWN',
                        'name': 'Unknown',
                        'confidence': 0,
                        'face_coords': (x, y, w, h)
                    }
                
                last_recognition_time = current_time
                show_result_until = current_time + datetime.timedelta(seconds=2)
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # Display info on frame
            info_text = f"Faces Detected: {len(faces)}"
            cv2.putText(display_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.WHITE, 2)
            
            if len(faces) > 0 and not show_result_until:
                cooldown_remaining = max(0, recognition_cooldown - time_since_last)
                if cooldown_remaining > 0:
                    cv2.putText(display_frame, f"Next scan: {cooldown_remaining:.1f}s", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.YELLOW, 2)
            
            # Instructions
            cv2.putText(display_frame, "Press 'q' to quit", 
                       (10, display_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.WHITE, 1)
            
            # Show frame
            cv2.imshow('Bus Access System', display_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Exiting...")
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Show summary
        self.show_summary()
    
    def show_summary(self):
        """Show session summary"""
        print("\n" + "=" * 60)
        print("📊 SESSION SUMMARY")
        print("=" * 60)
        
        total_logs = len(self.db['access_logs'])
        allowed = sum(1 for log in self.db['access_logs'] if log['status'] == 'allowed')
        denied = sum(1 for log in self.db['access_logs'] if log['status'] == 'denied_unpaid')
        unknown = sum(1 for log in self.db['access_logs'] if log['status'] == 'unrecognized')
        
        print(f"Total Scans: {total_logs}")
        print(f"✅ Allowed (Paid): {allowed}")
        print(f"❌ Denied (Unpaid): {denied}")
        print(f"❓ Unknown: {unknown}")
        print(f"\n📸 Total captures saved: {len(self.db['unpaid_captures'])}")
        print(f"📁 Location: data/unpaid_captures/")
        print("=" * 60)


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("🚌 SMART BUS ACCESS SYSTEM")
    print("=" * 60)
    
    # You can change the minimum confidence here (default: 70%)
    MIN_CONFIDENCE = 70.0  # Only accept if confidence >= 70%
    
    system = LiveBusAccessSystem(min_confidence=MIN_CONFIDENCE)
    
    print("\n📋 Database records:")
    for student_id, data in system.students.items():
        status = "✅ PAID" if data['fee_status'] == 'paid' else "❌ UNPAID"
        print(f"  {student_id}: {data['name']} - {status}")
    
    print("\n" + "=" * 60)
    print("🎥 Starting camera in 3 seconds...")
    print("=" * 60)
    
    # Auto-start after 3 seconds
    import time
    time.sleep(3)
    
    system.start_live_system()


if __name__ == "__main__":
    main()