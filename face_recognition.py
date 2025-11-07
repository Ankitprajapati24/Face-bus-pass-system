"""
Face Recognition Core Module
Handles all face detection and recognition operations
"""

import cv2
import os
import numpy as np
from deepface import DeepFace
import pickle
from pathlib import Path
import json
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self, registered_faces_dir='data/registered_faces', 
                 model_name='Facenet512', threshold=0.6):
        """
        Initialize Face Recognition System
        
        Args:
            registered_faces_dir: Directory containing registered face images
            model_name: DeepFace model ('VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib')
            threshold: Similarity threshold (lower = stricter matching)
        """
        self.registered_faces_dir = Path(registered_faces_dir)
        self.registered_faces_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.threshold = threshold
        
        # Cache for face embeddings (for faster recognition)
        self.embeddings_cache_file = 'data/face_embeddings_cache.pkl'
        self.embeddings_cache = {}
        
        # Face detection cascade (for quick face detection)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print(f"🚀 Face Recognition System initialized")
        print(f"   Model: {model_name}")
        print(f"   Threshold: {threshold}")
        print(f"   Registered faces directory: {self.registered_faces_dir}")
        
        # Load existing embeddings cache
        self.load_embeddings_cache()
    
    def detect_faces(self, image):
        """
        Quick face detection using Haar Cascade
        
        Args:
            image: Image array (BGR format)
            
        Returns:
            List of face rectangles [(x, y, w, h), ...]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces
    
    def extract_face_embedding(self, image_path):
        """
        Extract face embedding from image using DeepFace
        
        Args:
            image_path: Path to image file
            
        Returns:
            embedding: Face embedding vector
        """
        try:
            # Use DeepFace to extract embedding
            embedding_objs = DeepFace.represent(
                img_path=str(image_path),
                model_name=self.model_name,
                enforce_detection=False
            )
            
            if embedding_objs and len(embedding_objs) > 0:
                return np.array(embedding_objs[0]['embedding'])
            return None
            
        except Exception as e:
            print(f"Error extracting embedding from {image_path}: {e}")
            return None
    
    def register_face(self, student_id, name, image_path, department=''):
        """
        Register a new student face
        
        Args:
            student_id: Unique student ID
            name: Student name
            image_path: Path to student's face image
            department: Student's department
            
        Returns:
            bool: Success status
        """
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                return False
            
            # Read and validate image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Failed to read image: {image_path}")
                return False
            
            # Detect face in image
            faces = self.detect_faces(image)
            if len(faces) == 0:
                print(f"❌ No face detected in image")
                return False
            
            if len(faces) > 1:
                print(f"⚠️  Multiple faces detected, using first face")
            
            # Extract face region
            x, y, w, h = faces[0]
            face_image = image[y:y+h, x:x+w]
            
            # Save registered face
            filename = f"{student_id}_{name.replace(' ', '_')}.jpg"
            save_path = self.registered_faces_dir / filename
            cv2.imwrite(str(save_path), face_image)
            
            # Extract and cache embedding
            embedding = self.extract_face_embedding(save_path)
            if embedding is not None:
                self.embeddings_cache[student_id] = {
                    'embedding': embedding,
                    'name': name,
                    'department': department,
                    'image_path': str(save_path),
                    'registered_date': datetime.now().isoformat()
                }
                self.save_embeddings_cache()
                
                print(f"✅ Successfully registered: {name} (ID: {student_id})")
                return True
            else:
                print(f"❌ Failed to extract embedding")
                return False
                
        except Exception as e:
            print(f"❌ Error registering face: {e}")
            return False
    
    def recognize_face(self, image_path, return_all_matches=False):
        """
        Recognize face from image
        
        Args:
            image_path: Path to captured image or image array
            return_all_matches: If True, return all matches with scores
            
        Returns:
            dict: Recognition result with student_id, confidence, etc.
        """
        try:
            # Handle both file path and numpy array
            if isinstance(image_path, str):
                if not os.path.exists(image_path):
                    return {'status': 'error', 'message': 'Image not found'}
                image = cv2.imread(image_path)
            else:
                image = image_path
            
            if image is None:
                return {'status': 'error', 'message': 'Failed to read image'}
            
            # Quick face detection
            faces = self.detect_faces(image)
            if len(faces) == 0:
                return {
                    'status': 'no_face',
                    'message': 'No face detected in image'
                }
            
            # Save temporary image for DeepFace
            temp_path = 'data/temp_capture.jpg'
            cv2.imwrite(temp_path, image)
            
            # Get embedding for captured face
            captured_embedding = self.extract_face_embedding(temp_path)
            
            if captured_embedding is None:
                return {
                    'status': 'error',
                    'message': 'Failed to extract face features'
                }
            
            # Compare with all registered faces
            matches = []
            
            for student_id, data in self.embeddings_cache.items():
                registered_embedding = data['embedding']
                
                # Calculate cosine similarity
                similarity = self.cosine_similarity(
                    captured_embedding, 
                    registered_embedding
                )
                
                # Convert to distance (lower is better)
                distance = 1 - similarity
                
                if distance < self.threshold:
                    confidence = (1 - distance) * 100
                    matches.append({
                        'student_id': student_id,
                        'name': data['name'],
                        'department': data.get('department', ''),
                        'confidence': round(confidence, 2),
                        'distance': round(distance, 4)
                    })
            
            # Sort by confidence (highest first)
            matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if len(matches) == 0:
                return {
                    'status': 'unknown',
                    'message': 'Face not recognized',
                    'confidence': 0
                }
            
            if return_all_matches:
                return {
                    'status': 'recognized',
                    'matches': matches
                }
            else:
                # Return best match
                best_match = matches[0]
                return {
                    'status': 'recognized',
                    'student_id': best_match['student_id'],
                    'name': best_match['name'],
                    'department': best_match['department'],
                    'confidence': best_match['confidence'],
                    'all_matches': len(matches)
                }
                
        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)
    
    def load_embeddings_cache(self):
        """Load cached embeddings from file"""
        if os.path.exists(self.embeddings_cache_file):
            try:
                with open(self.embeddings_cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                print(f"✅ Loaded {len(self.embeddings_cache)} cached embeddings")
            except Exception as e:
                print(f"⚠️  Failed to load cache: {e}")
                self.embeddings_cache = {}
        else:
            print("ℹ️  No existing cache found, will create new cache")
    
    def save_embeddings_cache(self):
        """Save embeddings cache to file"""
        try:
            os.makedirs(os.path.dirname(self.embeddings_cache_file), exist_ok=True)
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            print(f"✅ Saved embeddings cache ({len(self.embeddings_cache)} entries)")
        except Exception as e:
            print(f"❌ Failed to save cache: {e}")
    
    def rebuild_cache(self):
        """Rebuild embeddings cache from registered faces directory"""
        print("🔄 Rebuilding embeddings cache...")
        self.embeddings_cache = {}
        
        image_files = list(self.registered_faces_dir.glob('*.jpg')) + \
                      list(self.registered_faces_dir.glob('*.png'))
        
        for image_path in image_files:
            try:
                # Extract student_id from filename (format: studentid_name.jpg)
                filename = image_path.stem
                parts = filename.split('_', 1)
                
                if len(parts) >= 2:
                    student_id = parts[0]
                    name = parts[1].replace('_', ' ')
                    
                    print(f"Processing: {student_id} - {name}")
                    
                    embedding = self.extract_face_embedding(image_path)
                    if embedding is not None:
                        self.embeddings_cache[student_id] = {
                            'embedding': embedding,
                            'name': name,
                            'department': '',
                            'image_path': str(image_path),
                            'registered_date': datetime.now().isoformat()
                        }
            except Exception as e:
                print(f"⚠️  Error processing {image_path}: {e}")
        
        self.save_embeddings_cache()
        print(f"✅ Cache rebuilt with {len(self.embeddings_cache)} entries")
    
    def get_registered_students(self):
        """Get list of all registered students"""
        students = []
        for student_id, data in self.embeddings_cache.items():
            students.append({
                'student_id': student_id,
                'name': data['name'],
                'department': data.get('department', ''),
                'registered_date': data.get('registered_date', '')
            })
        return students
    
    def delete_student(self, student_id):
        """Remove a student from the system"""
        if student_id in self.embeddings_cache:
            # Delete image file
            image_path = self.embeddings_cache[student_id].get('image_path')
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
            
            # Remove from cache
            del self.embeddings_cache[student_id]
            self.save_embeddings_cache()
            
            print(f"✅ Deleted student: {student_id}")
            return True
        return False


# Example usage and testing
if __name__ == "__main__":
    print("=" * 50)
    print("Face Recognition System - Test Mode")
    print("=" * 50)
    
    # Initialize system
    fr_system = FaceRecognitionSystem(model_name='Facenet512', threshold=0.6)
    
    # Check registered students
    students = fr_system.get_registered_students()
    print(f"\n📋 Registered Students: {len(students)}")
    for student in students:
        print(f"   - {student['student_id']}: {student['name']}")
    
    print("\n✅ System ready for testing!")
    print("\nAvailable commands:")
    print("  1. Register new face")
    print("  2. Recognize face from webcam")
    print("  3. Test with image file")
    print("  4. Rebuild cache")
    print("  5. Exit")