import os

class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev')
    
    # File upload settings
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Model settings
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'inswapper_128.onnx')
    
    # Processing settings
    MIN_FRAME_SIZE = (480, 360)
    TARGET_FPS = 30
    FRAME_TIMEOUT = 1.0 / TARGET_FPS
    FACE_CACHE_DURATION = 0.1
    
    # Debug settings
    DEBUG_FRAMES_DIR = 'debug_frames'
    SAVE_INTERVAL = 3  # seconds
    
    # Black rectangle settings
    BLACK_RECTANGLE_ENABLED = True
    BLACK_RECTANGLE_POSITION = (220, 120)
    BLACK_RECTANGLE_SIZE = (200, 200) 