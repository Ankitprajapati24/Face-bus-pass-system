"""
Flask Web API for Bus Access System
COMPLETE WORKING VERSION
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
from datetime import datetime
from pathlib import Path
import sys
import subprocess
import cv2
import numpy as np
import base64 


# ==================== PATH SETUP ====================
CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent.parent
sys.path.insert(0, str(CURRENT_FILE.parent))

print(f"📁 Project Root: {BASE_DIR}")
print(f"📁 Backend Dir: {CURRENT_FILE.parent}")

# Import face recognition
try:
    from face_recognition import FaceRecognitionSystem
except ImportError as e:
    FaceRecognitionSystem = None
    print(f"⚠️  Face recognition not available: {e}")

app = Flask(__name__)
CORS(app)

# Configuration
DATABASE_FILE = BASE_DIR / 'students_database.json'
fr_system = None
scanner_process = None

# Initialize face recognition
try:
    if FaceRecognitionSystem:
        original_dir = os.getcwd()
        os.chdir(BASE_DIR)
        fr_system = FaceRecognitionSystem()
        os.chdir(original_dir)
except Exception as e:
    print(f"⚠️  Could not initialize face recognition: {e}")

# Create directories
(BASE_DIR / 'data' / 'group_scans').mkdir(parents=True, exist_ok=True)
(BASE_DIR / 'data' / 'registered_faces').mkdir(parents=True, exist_ok=True)
(BASE_DIR / 'data' / 'unpaid_captures').mkdir(parents=True, exist_ok=True)

# ==================== DATABASE ====================

def load_database():
    if DATABASE_FILE.exists():
        try:
            with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {'students': {}, 'access_logs': [], 'unpaid_captures': []}
    return {'students': {}, 'access_logs': [], 'unpaid_captures': []}

def save_database(db):
    try:
        with open(DATABASE_FILE, 'w', encoding='utf-8') as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving database: {e}")

# ==================== ROUTES ====================

@app.route('/')
def index():
    try:
        return send_from_directory(BASE_DIR / 'frontend', 'dashboard.html')
    except Exception as e:
        return f"Error: {e}", 500


@app.route('/<path:filename>')
def serve_static_files(filename):
    """Serves all other static files (like script.js) from the frontend directory."""
    try:
        return send_from_directory(BASE_DIR / 'frontend', filename)
    except Exception as e:
        return f"File not found: {filename}", 404

@app.route('/api/status')
def get_status():
    try:
        db = load_database()
        total_students = len(db['students'])
        paid_students = sum(1 for s in db['students'].values() if s.get('fee_status') == 'paid')
        unpaid_students = total_students - paid_students
        
        today = datetime.now().strftime('%Y-%m-%d')
        today_logs = [log for log in db['access_logs'] if log.get('timestamp', '').startswith(today)]
        
        total_access = len(today_logs)
        denied_access = sum(1 for log in today_logs if log.get('status', '').startswith('denied') or log.get('status') == 'unrecognized')
        
        return jsonify({
            'status': 'online',
            'total_students': total_students,
            'paid_students': paid_students,
            'unpaid_students': unpaid_students,
            'total_access_today': total_access,
            'denied_today': denied_access,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students')
def get_students():
    try:
        db = load_database()
        students_list = []
        for student_id, data in db['students'].items():
            students_list.append({
                'id': student_id,
                'name': data['name'],
                'roll_number': data.get('roll_number', ''),
                'department': data.get('department', ''),
                'fee_status': data.get('fee_status', 'unpaid'),
                'face_registered': data.get('face_registered', False)
            })
        return jsonify({'students': students_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs')
def get_logs():
    try:
        db = load_database()
        logs = db['access_logs'][-50:]
        logs.reverse()
        return jsonify({'logs': logs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs/today')
def get_today_logs():
    try:
        db = load_database()
        today = datetime.now().strftime('%Y-%m-%d')
        today_logs = [log for log in db['access_logs'] if log.get('timestamp', '').startswith(today)]
        today_logs.reverse()
        return jsonify({'logs': today_logs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== WEB RECOGNITION API ====================
# THIS IS THE NEW ENDPOINT FOR THE WEB-BASED SCANNER

@app.route('/api/recognize', methods=['POST'])
def web_recognize():
    
    # 1. Check if Face Recognition system is loaded
    if not fr_system:
        return jsonify({
            'status': 'ERROR',
            'message': 'Face Recognition system is not initialized on the server.'
        }), 500

    # 2. Get the image from the POST request
    try:
        data = request.json
        if 'image_base64' not in data:
            return jsonify({'status': 'ERROR', 'message': 'Missing image_base64 field.'}), 400
        
        # Get the Base64 string and decode it
        image_data = data['image_base64'].split(',')[1] # Remove "data:image/jpeg;base64,"
        image_bytes = base64.b64decode(image_data)
        
        # Convert bytes to a numpy array for OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    except Exception as e:
        print(f"Error decoding image: {e}")
        return jsonify({'status': 'ERROR', 'message': f'Error decoding image: {e}'}), 400
    
    # 3. Save to a temporary file for fr_system to read
    # (fr_system.recognize_face expects a file path)
    temp_image_path = BASE_DIR / 'data' / 'temp_web_capture.jpg'
    try:
        cv2.imwrite(str(temp_image_path), frame)
    except Exception as e:
        return jsonify({'status': 'ERROR', 'message': f'Could not save temp file: {e}'}), 500

    # 4. === RUN RECOGNITION ===
    try:
        # Use the same FaceRecognitionSystem loaded at startup
        result = fr_system.recognize_face(str(temp_image_path))
    except Exception as e:
        print(f"Recognition error: {e}")
        return jsonify({'status': 'ERROR', 'message': f'Recognition error: {e}'}), 500

    # 5. === CHECK DATABASE FOR FEE STATUS ===
    try:
        db = load_database()
        students = db.get('students', {})
        
        if result['status'] == 'recognized':
            student_id = result['student_id']
            name = result['name']
            confidence = result['confidence']
            
            if student_id in students:
                student_data = students[student_id]
                fee_status = student_data.get('fee_status', 'unpaid')
                
                if fee_status == 'paid':
                 # SUCCESS: Recognized AND Paid
                    return jsonify({
                        'status': 'ALLOWED',
                        'name': name,
                        'student_id': student_id,
                        'confidence': confidence,
                        'face_coords': result.get('face_coords') # <-- ADD THIS
                    })
                else:
               # FAILURE: Recognized but Unpaid
                    return jsonify({
                        'status': 'DENIED',
                        'name': name,
                        'student_id': student_id,
                        'message': 'Fee status is UNPAID.',
                        'confidence': confidence,
                        'face_coords': result.get('face_coords') # <-- ADD THIS
                    })
                # FAILURE: Face matched but not in student database
                return jsonify({
                    'status': 'DENIED',
                    'name': name,
                    'student_id': student_id,
                    'message': 'Face is registered but student data is missing from database.',
                    'confidence': confidence
                })
        
        elif result['status'] == 'unknown':
            # FAILURE: Not recognized
            return jsonify({
                'status': 'UNKNOWN',
                'message': 'Person is not registered in the system.',
                'face_coords': result.get('face_coords')
            })
            
        else:
            # FAILURE: No face found or other error
            return jsonify({
                'status': 'ERROR',
                'message': result.get('message', 'No face detected or system error.')
            })

    except Exception as e:
        print(f"Error checking database: {e}")
        return jsonify({'status': 'ERROR', 'message': f'Database processing error: {e}'}), 500

    finally:
        # 6. Clean up the temporary file
        if temp_image_path.exists():
            os.remove(temp_image_path)


# ==================== SCANNER CONTROL ====================

@app.route('/api/scanner/start', methods=['POST'])
def start_scanner():
    global scanner_process
    try:
        if scanner_process and scanner_process.poll() is None:
            return jsonify({'status': 'already_running', 'message': 'Scanner already running'})
        
        scanner_path = BASE_DIR / 'live_group_scanner.py'
        if not scanner_path.exists():
            return jsonify({'status': 'error', 'message': f'Scanner not found at {scanner_path}'}), 404
        
        scanner_process = subprocess.Popen(
            [sys.executable, str(scanner_path)],
            cwd=str(BASE_DIR)
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Scanner started',
            'pid': scanner_process.pid
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/scanner/stop', methods=['POST'])
def stop_scanner():
    global scanner_process
    try:
        if scanner_process and scanner_process.poll() is None:
            scanner_process.terminate()
            scanner_process.wait(timeout=5)
            scanner_process = None
            return jsonify({'status': 'success', 'message': 'Scanner stopped'})
        return jsonify({'status': 'not_running', 'message': 'Scanner not running'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/scanner/status')
def scanner_status():
    global scanner_process
    is_running = scanner_process and scanner_process.poll() is None
    return jsonify({
        'status': 'running' if is_running else 'stopped',
        'pid': scanner_process.pid if is_running else None
    })

@app.route('/api/scans')
def get_scans():
    try:
        scans_dir = BASE_DIR / 'data' / 'group_scans'
        if not scans_dir.exists():
            return jsonify({'scans': []})
        
        scans = []
        for img_file in sorted(scans_dir.glob('*.jpg'), reverse=True):
            txt_file = img_file.with_suffix('.txt')
            
            scan_info = {
                'image': img_file.name,
                'timestamp': datetime.fromtimestamp(img_file.stat().st_mtime).isoformat(),
                'size': img_file.stat().st_size
            }
            
            if txt_file.exists():
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        scan_info['report'] = f.read()
                except:
                    pass
            
            scans.append(scan_info)
        
        return jsonify({'scans': scans[:20]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/scans/<filename>')
def get_scan_image(filename):
    try:
        return send_from_directory(BASE_DIR / 'data' / 'group_scans', filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/images/<path:filename>')
def serve_image(filename):
    try:
        return send_from_directory(BASE_DIR / 'data' / 'unpaid_captures', filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

# ==================== HEALTH CHECK ====================

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'face_recognition': fr_system is not None
    })

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚌 BUS ACCESS SYSTEM - WEB SERVER")
    print("=" * 60)
    print(f"\n📡 Server starting from: {CURRENT_FILE.parent}")
    print(f"📁 Project root: {BASE_DIR}")
    print(f"📁 Database: {DATABASE_FILE}")
    print(f"🤖 Face Recognition: {'✅ Available' if fr_system else '❌ Not Available'}")
    print("\n🌐 Access dashboard at: http://localhost:5000")
    print("📱 For remote access: Use your local IP address")
    print("\n" + "=" * 60 + "\n")
    
    app.run()