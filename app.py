"""
Flask Web Application for Real-Time Football Player Tracking
Provides MJPEG streaming with live jersey number recognition.
"""

import os
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from werkzeug.utils import secure_filename
from camera import VideoCamera

app = Flask(__name__)
app.config['SECRET_KEY'] = 'futbl-tracking-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Global camera instance
camera = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_camera():
    """Get the current camera instance."""
    global camera
    return camera


def set_camera(video_path):
    """Initialize camera with a new video."""
    global camera
    if camera:
        camera.release()
    camera = VideoCamera(video_path)
    return camera


def generate_frames():
    """Generator function for MJPEG streaming."""
    cam = get_camera()
    if cam is None:
        return

    while True:
        frame = cam.get_frame()

        if frame is None:
            # Video ended or error
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    """Upload page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, webm'}), 400

    # Save file
    filename = secure_filename(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Initialize camera with uploaded video
        set_camera(filepath)
        return redirect(url_for('dashboard'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dashboard')
def dashboard():
    """Dashboard page with video stream and stats."""
    if get_camera() is None:
        return redirect(url_for('index'))
    return render_template('dashboard.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG video stream endpoint."""
    if get_camera() is None:
        return "No video loaded", 404

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/status')
def status():
    """JSON endpoint for current frame count and player data."""
    cam = get_camera()
    if cam is None:
        return jsonify({
            'current_frame': 0,
            'total_frames': 0,
            'fps': 0,
            'is_playing': False,
            'players': []
        })

    return jsonify(cam.get_stats())


@app.route('/toggle_play', methods=['POST'])
def toggle_play():
    """Toggle play/pause state."""
    cam = get_camera()
    if cam is None:
        return jsonify({'error': 'No video loaded'}), 404

    is_playing = cam.toggle_play()
    return jsonify({'is_playing': is_playing})


@app.route('/seek', methods=['POST'])
def seek():
    """Seek to a specific frame."""
    cam = get_camera()
    if cam is None:
        return jsonify({'error': 'No video loaded'}), 404

    data = request.get_json()
    if not data or 'frame' not in data:
        return jsonify({'error': 'Frame number required'}), 400

    frame_number = int(data['frame'])
    success = cam.seek_to_frame(frame_number)

    return jsonify({
        'success': success,
        'current_frame': cam.current_frame
    })


@app.route('/get_frame')
def get_frame():
    """Get a single frame (used when paused/seeking)."""
    cam = get_camera()
    if cam is None:
        return "No video loaded", 404

    frame_bytes = cam.get_single_frame()
    if frame_bytes is None:
        return "Could not get frame", 500

    return Response(frame_bytes, mimetype='image/jpeg')


@app.route('/reset', methods=['POST'])
def reset():
    """Reset and go back to upload page."""
    global camera
    if camera:
        camera.release()
        camera = None
    return jsonify({'success': True})


if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs('uploads', exist_ok=True)

    print("=" * 60)
    print("Football Player Tracking Application")
    print("=" * 60)
    print("Starting server on http://127.0.0.1:5000")
    print("Upload a video to begin tracking...")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
