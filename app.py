from flask import Flask, render_template, request, redirect, url_for, Response
from flask_sock import Sock
import cv2
import os
import threading
import numpy as np
from modules.processors.frame.face_swapper import process_frame
from modules.face_analyser import get_one_face
import io
from PIL import Image
import onnxruntime as ort
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
sock = Sock(app)

# Cloud deployment configurations
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Global variables
uploaded_face_path = None
recording = False
out = None  # Video writer object
source_face = None
execution_provider = None
session = None

# Add request logging
@app.before_request
def log_request_info():
    if request.path == '/upload_frame':
        logger.warning(f"Unexpected request to /upload_frame from {request.remote_addr} - User-Agent: {request.user_agent.string}")

# Add handler for /upload_frame to properly respond
@app.route('/upload_frame', methods=['POST'])
def handle_upload_frame():
    logger.warning(f"Received upload_frame request from {request.remote_addr}")
    return "This endpoint is not supported. Please use WebSocket for frame processing.", 405

def get_onnx_session(provider):
    """Create ONNX session with specified execution provider"""
    providers = {
        'CPUExecutionProvider': ['CPUExecutionProvider'],
        'CUDAExecutionProvider': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        'OpenVINOExecutionProvider': ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
    }
    
    try:
        # Use absolute path for the model file
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inswapper_128.onnx')
        logger.info(f"Attempting to load model from: {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            return None
            
        # Get available providers
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX providers: {available_providers}")
        
        # Create session with specified provider
        session = ort.InferenceSession(model_path, providers=providers.get(provider, ['CPUExecutionProvider']))
        logger.info(f"Successfully created ONNX session with provider: {provider}")
        return session
    except Exception as e:
        logger.error(f"Error creating ONNX session: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        return None

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        return "An error occurred while loading the page. Please check the server logs.", 500

@app.route('/upload_face', methods=['POST'])
def upload_face():
    global uploaded_face_path, source_face, execution_provider, session
    
    if 'face' not in request.files:
        return "No file part", 400
    
    file = request.files['face']
    if file.filename == '':
        return "No selected file", 400
    
    # Get execution provider
    execution_provider = request.form.get('provider', 'CPUExecutionProvider')
    logger.info(f"Selected execution provider: {execution_provider}")
    
    try:
        # Save uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        uploaded_face_path = file_path
        
        # Initialize ONNX session
        session = get_onnx_session(execution_provider)
        if session is None:
            return "Error initializing face processing", 500
        
        # Pre-process the source face
        source_face = get_one_face(cv2.imread(uploaded_face_path))
        if source_face is None:
            return "No face detected in the image", 400
            
        logger.info("Face image processed successfully")
        return "Success", 200
        
    except Exception as e:
        logger.error(f"Error processing face image: {e}")
        return str(e), 500

@sock.route('/ws')
def ws(ws):
    global source_face, recording, out, session
    
    if source_face is None:
        ws.send("No face image uploaded. Please upload a face image first.")
        return

    if session is None:
        ws.send("Error: Face processing not initialized. Please try uploading the face image again.")
        return

    try:
        while True:
            try:
                # Receive frame from client
                frame_data = ws.receive()
                if frame_data is None:
                    break

                # Convert received data to OpenCV image
                frame = np.array(Image.open(io.BytesIO(frame_data)))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Process the frame
                processed_frame = process_frame(source_face, frame, session)

                if recording and out:
                    out.write(processed_frame)

                # Convert processed frame back to JPEG
                _, buffer = cv2.imencode('.jpg', processed_frame)
                processed_frame_data = buffer.tobytes()

                # Send processed frame back to client
                ws.send(processed_frame_data)

            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                ws.send(f"Error processing frame: {str(e)}")
                time.sleep(0.1)  # Prevent tight error loop
                continue

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws.send(f"WebSocket error: {str(e)}")

@app.route('/record/start', methods=['POST'])
def start_record():
    global recording, out
    try:
        recording = True
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'output_{timestamp}.mp4'
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'X264'), 20.0, (640, 480))
        logger.info(f"Recording started: {filename}")
        return '', 204
    except Exception as e:
        logger.error(f"Error starting recording: {e}")
        return str(e), 500

@app.route('/record/stop', methods=['POST'])
def stop_record():
    global recording, out
    try:
        recording = False
        if out:
            out.release()
            out = None
        logger.info("Recording stopped")
        return '', 204
    except Exception as e:
        logger.error(f"Error stopping recording: {e}")
        return str(e), 500

if __name__ == '__main__':
    # For cloud deployment, use the following:
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'  # Listen on all network interfaces
    app.run(host=host, port=port, debug=False)  # Set debug=False for production
