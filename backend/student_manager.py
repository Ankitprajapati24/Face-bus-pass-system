"""
Student Manager - Complete Registration, Update, and Delete Tool
Register faces AND add to database in one go!
"""

import cv2
import json
import os
from pathlib import Path
from datetime import datetime
from face_recognition import FaceRecognitionSystem

class StudentManager:
    def __init__(self):
        self.fr_system = FaceRecognitionSystem()
        self.database_file = 'students_database.json'  # THIS is your file name
        self.load_database()
        
        print("✅ Student Manager Initialized!")
        print(f"📁 Database file: {self.database_file}")
    
    def load_database(self):
        """Load database from students_database.json"""
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'r', encoding='utf-8') as f:
                    self.db = json.load(f)
                print(f"✅ Loaded database: {len(self.db.get('students', {}))} students")
            except Exception as e:
                print(f"⚠️  Error loading database: {e}")
                print("Creating new database...")
                self.db = {
                    'students': {},
                    'access_logs': [],
                    'unpaid_captures': []
                }
        else:
            print("⚠️  No database file found. Creating new one...")
            self.db = {
                'students': {},
                'access_logs': [],
                'unpaid_captures': []
            }
        
        self.students = self.db.get('students', {})
    
    def save_database(self):
        """Save database to students_database.json"""
        try:
            # Update students in database
            self.db['students'] = self.students
            
            # Save with proper formatting
            with open(self.database_file, 'w', encoding='utf-8') as f:
                json.dump(self.db, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Database saved to {self.database_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving database: {e}")
            return False
    
    def register_with_webcam(self):
        """Register student using webcam"""
        print("\n" + "=" * 60)
        print("📸 REGISTER NEW STUDENT")
        print("=" * 60)
        
        # Get details
        student_id = input("\nStudent ID (e.g., STU002): ").strip()
        if not student_id:
            print("❌ Student ID required!")
            return
        
        # Check if already exists
        if student_id in self.students:
            print(f"⚠️  Student {student_id} already exists!")
            overwrite = input("Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                return
        
        name = input("Full Name: ").strip()
        if not name:
            print("❌ Name required!")
            return
        
        roll_number = input("Roll Number: ").strip()
        department = input("Department: ").strip()
        
        print("\nFee Status Options:")
        print("  1. Paid")
        print("  2. Unpaid")
        fee_choice = input("Select (1-2): ").strip()
        fee_status = 'paid' if fee_choice == '1' else 'unpaid'
        
        print(f"\n📋 Summary:")
        print(f"   ID: {student_id}")
        print(f"   Name: {name}")
        print(f"   Roll: {roll_number}")
        print(f"   Dept: {department}")
        print(f"   Fee: {fee_status.upper()}")
        
        confirm = input("\nProceed? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Cancelled!")
            return
        
        # Capture face
        print("\n🎥 Opening camera...")
        print("Position your face and press SPACE to capture")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera!")
            return
        
        captured = False
        temp_path = 'data/temp_registration.jpg'
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = self.fr_system.detect_faces(frame)
            
            display = frame.copy()
            for (x, y, w, h) in faces:
                color = (0, 255, 0) if len(faces) == 1 else (0, 165, 255)
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
            
            # Instructions
            if len(faces) == 0:
                cv2.putText(display, "No face detected!", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif len(faces) == 1:
                cv2.putText(display, "Ready! Press SPACE", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Multiple faces! Show only one", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            cv2.putText(display, "ESC to cancel", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Register Student', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32 and len(faces) == 1:  # SPACE
                cv2.imwrite(temp_path, frame)
                captured = True
                print("✅ Image captured!")
                break
            elif key == 27:  # ESC
                print("❌ Cancelled!")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not captured:
            return
        
        # Register face
        print("\n🔍 Registering face...")
        success = self.fr_system.register_face(
            student_id=student_id,
            name=name,
            image_path=temp_path,
            department=department
        )
        
        if not success:
            print("❌ Face registration failed!")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return
        
        # Add to database - THIS AUTOMATICALLY UPDATES students_database.json
        print("💾 Adding to database...")
        self.students[student_id] = {
            'name': name,
            'roll_number': roll_number,
            'department': department,
            'email': f"{roll_number.lower()}@college.edu" if roll_number else '',
            'phone': '',
            'fee_status': fee_status,
            'fee_paid_date': datetime.now().strftime('%Y-%m-%d') if fee_status == 'paid' else '',
            'valid_until': '2025-12-31' if fee_status == 'paid' else '',
            'face_registered': True,
            'registered_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # SAVE TO students_database.json FILE
        if self.save_database():
            print("✅ Database file updated successfully!")
        else:
            print("❌ Warning: Database save failed!")
            return
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        print("\n" + "=" * 60)
        print("🎉 REGISTRATION SUCCESS!")
        print("=" * 60)
        print(f"✅ {name} registered successfully!")
        print(f"✅ Face saved: data/registered_faces/{student_id}_{name.replace(' ', '_')}.jpg")
        print(f"✅ Database updated: {self.database_file}")
        print(f"✅ Fee status: {fee_status.upper()}")
        print("\n💡 You can now use live_group_scanner.py to detect this student!")
        print("=" * 60)
    
    def list_students(self):
        """List all registered students"""
        print("\n" + "=" * 60)
        print(f"📋 REGISTERED STUDENTS ({len(self.students)})")
        print("=" * 60)
        
        if not self.students:
            print("No students registered yet!")
            return
        
        for i, (student_id, data) in enumerate(self.students.items(), 1):
            status = "✅ PAID" if data.get('fee_status') == 'paid' else "❌ UNPAID"
            print(f"\n{i}. {data['name']}")
            print(f"   ID: {student_id}")
            print(f"   Roll: {data.get('roll_number', 'N/A')}")
            print(f"   Dept: {data.get('department', 'N/A')}")
            print(f"   Fee: {status}")
    
    def update_fee_status(self):
        """Update student fee status"""
        print("\n" + "=" * 60)
        print("💰 UPDATE FEE STATUS")
        print("=" * 60)
        
        if not self.students:
            print("No students registered!")
            return
        
        # Show list
        print("\nRegistered Students:")
        for i, (sid, data) in enumerate(self.students.items(), 1):
            status = "PAID" if data.get('fee_status') == 'paid' else "UNPAID"
            print(f"{i}. {sid}: {data['name']} - {status}")
        
        student_id = input("\nEnter Student ID to update: ").strip()
        
        if student_id not in self.students:
            print(f"❌ Student {student_id} not found!")
            return
        
        print(f"\nUpdating: {self.students[student_id]['name']}")
        print("1. Mark as PAID")
        print("2. Mark as UNPAID")
        
        choice = input("Select (1-2): ").strip()
        
        if choice == '1':
            self.students[student_id]['fee_status'] = 'paid'
            self.students[student_id]['fee_paid_date'] = datetime.now().strftime('%Y-%m-%d')
            self.students[student_id]['valid_until'] = '2025-12-31'
            print(f"✅ {self.students[student_id]['name']} marked as PAID")
        elif choice == '2':
            self.students[student_id]['fee_status'] = 'unpaid'
            self.students[student_id]['fee_paid_date'] = ''
            self.students[student_id]['valid_until'] = ''
            print(f"❌ {self.students[student_id]['name']} marked as UNPAID")
        else:
            print("Invalid choice!")
            return
        
        self.save_database()
    
    def delete_student(self):
        """Delete a student"""
        print("\n" + "=" * 60)
        print("🗑️  DELETE STUDENT")
        print("=" * 60)
        
        if not self.students:
            print("No students registered!")
            return
        
        # Show list
        print("\nRegistered Students:")
        for i, (sid, data) in enumerate(self.students.items(), 1):
            print(f"{i}. {sid}: {data['name']}")
        
        student_id = input("\nEnter Student ID to delete: ").strip()
        
        if student_id not in self.students:
            print(f"❌ Student {student_id} not found!")
            return
        
        name = self.students[student_id]['name']
        
        print(f"\n⚠️  WARNING: This will delete:")
        print(f"   • {name} from database")
        print(f"   • Face image from registered_faces/")
        print(f"   • Cached embeddings")
        
        confirm = input("\nAre you sure? Type 'DELETE' to confirm: ").strip()
        
        if confirm != 'DELETE':
            print("❌ Cancelled!")
            return
        
        # Delete from database
        del self.students[student_id]
        self.save_database()
        
        # Delete face recognition data
        success = self.fr_system.delete_student(student_id)
        
        if success:
            print(f"\n✅ {name} deleted successfully!")
        else:
            print(f"\n⚠️  Database updated but face files not found")
    
    def main_menu(self):
        """Main menu"""
        while True:
            print("\n" + "=" * 60)
            print("🎓 STUDENT MANAGER")
            print("=" * 60)
            print("\n1. 📸 Register New Student (Webcam)")
            print("2. 📋 List All Students")
            print("3. 💰 Update Fee Status")
            print("4. 🗑️  Delete Student")
            print("5. ❌ Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                self.register_with_webcam()
            elif choice == '2':
                self.list_students()
            elif choice == '3':
                self.update_fee_status()
            elif choice == '4':
                self.delete_student()
            elif choice == '5':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid option!")
            
            input("\nPress ENTER to continue...")


def main():
    manager = StudentManager()
    manager.main_menu()


if __name__ == "__main__":
    main()
    