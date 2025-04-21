from flask import Flask
from app.routes import main_routes, image_routes, video_routes
from app.utils.config import Config
from app.services.model_service import preload_models

def create_app():
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(Config)
    
    # Register blueprints
    app.register_blueprint(main_routes.bp)
    app.register_blueprint(image_routes.bp)
    app.register_blueprint(video_routes.bp)
    
    # Preload models
    preload_models()
    
    return app 