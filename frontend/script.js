/*
 * Main JavaScript for the Bus Access System Dashboard
 */
document.addEventListener('DOMContentLoaded', () => {

    // === Dashboard Stats & Logs ===
    // (We will add functions here to load stats, students, and logs)
    
    
    // === Live Camera Scanner ===
    const video = document.getElementById('webcam-feed');
    const scanButton = document.getElementById('scan-button');
    const scanResult = document.getElementById('scan-result');
    const espStreamInput = document.getElementById('esp-stream-url');
    const useEspButton = document.getElementById('use-esp-stream');
    const useWebcamButton = document.getElementById('use-webcam');

    // ... (keep the other variables)

    const canvas = document.getElementById('overlay-canvas'); // <-- ADD THIS
    const ctx = canvas.getContext('2d'); // <-- ADD THIS

    
    // ... (rest of variables)

    let currentStream = null;

    // Function to start the webcam
    async function startWebcam() {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
        }
        
        
        try {
            currentStream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            video.srcObject = currentStream;
            video.style.display = 'block'; // Show video element
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth; 
                canvas.height = video.videoHeight;
            };
            scanButton.disabled = false;
            scanResult.innerHTML = '<p class="text-gray-500">Click "Scan Now" to check access</p>';
        } catch (err) {
            console.error("Error accessing webcam:", err);
            scanResult.innerHTML = `<p class="text-red-500">Error: Could not access webcam. ${err.message}</p>`;
        }
    }

    // Function to use the ESP32 stream
    // NOTE: This requires the ESP32 to be streaming on an accessible IP
    // and for the browser to not block mixed-content (HTTP on an HTTPS site)
    function startEspStream() {
        const url = espStreamInput.value;
        if (!url) {
            alert("Please enter the ESP32 Stream URL.");
            return;
        }
        
        // This is a simple way; a more robust way involves an MJPEG proxy
        // For now, we'll assume a simple image snapshot URL if it's not a video stream
        // This part is complex; let's focus on the webcam first.
        // For a true MJPEG stream, you'd set video.src = url
        
        alert("ESP32 streaming is complex due to browser security.\nWe will enable webcam scanning for now.");
        startWebcam();
    }

    // Event Listeners
    useWebcamButton.addEventListener('click', startWebcam);
    useEspButton.addEventListener('click', startEspStream);
    scanButton.addEventListener('click', captureAndRecognize);

    // Default to starting webcam
    startWebcam();


    // === The Core Recognition Function ===
    async function captureAndRecognize() {
        if (!video.srcObject) {
            alert("Camera feed is not active.");
            return;
        }

        // 1. Set UI to "Scanning..."
        scanButton.disabled = true;
        scanButton.innerText = "Scanning...";
        scanResult.innerHTML = '<p class="text-blue-500">Processing image...</p>';

        // 2. Capture frame
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // 3. Get image as Base64 data
        // We use 'jpeg' for smaller file size
        const base64Image = canvas.toDataURL('image/jpeg', 0.9);

        // 4. Send to our backend API
        try {
            const response = await fetch('/api/recognize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_base64: base64Image
                }),
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const data = await response.json();

            // 5. Display the result
            displayScanResult(data);

        } catch (err) {
            console.error("Error during recognition:", err);
            displayScanResult({ status: 'ERROR', message: err.message });
        } finally {
            // 6. Reset the button
            scanButton.disabled = false;
            scanButton.innerText = "Scan Now";
        }
    }

    function displayScanResult(data) {
        // Clear the canvas and the old text result
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        scanResult.innerHTML = '';

        // Get face coordinates, if they exist
        const coords = data.face_coords;

        if (coords) {
            // Coords are [x, y, w, h] from OpenCV
            const [x, y, w, h] = coords;
            
            let color = "#FF0000"; // Red (default for denied/unknown)
            let text = "ACCESS DENIED";
            let name = data.name || "Unknown";

            if (data.status === 'ALLOWED') {
                color = "#00FF00"; // Green
                text = "ACCESS GRANTED";
            } else if (data.status === 'DENIED') {
                text = "DENIED: UNPAID";
            } else if (data.status === 'UNKNOWN') {
                text = "UNKNOWN PERSON";
            }

            // --- Draw the Box ---
            ctx.strokeStyle = color;
            ctx.lineWidth = 4;
            ctx.strokeRect(x, y, w, h);

            // --- Draw the Text Background ---
            ctx.fillStyle = color;
            ctx.font = '20px Arial';
            const textWidth = ctx.measureText(text).width;
            const nameWidth = ctx.measureText(name).width;
            const bgWidth = Math.max(textWidth, nameWidth) + 20;
            ctx.fillRect(x, y - 50, bgWidth, 50);
            
            // --- Draw the Text ---
            ctx.fillStyle = "#000000"; // Black text
            ctx.fillText(text, x + 10, y - 28);
            ctx.font = '16px Arial';
            ctx.fillText(name, x + 10, y - 10);

        } else {
            // No face found, just show text message
            let html = '';
            if (data.status === 'ERROR') {
                html = `<div class..."text-red-500">${data.message}</div>`;
            } else {
                html = `<div class="text-yellow-500">No face was detected in the scan.</div>`;
            }
            scanResult.innerHTML = html;
        }
        
        // Automatically clear the box after 3 seconds
        setTimeout(() => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }, 3000);
    }
});