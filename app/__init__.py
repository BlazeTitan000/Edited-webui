from flask import Flask
from app.routes import main_routes, image_routes, video_routes
from app.utils.config import Config
from app.services.model_service import preload_models
import os

def create_app():
    app = Flask(__name__,
                template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
                static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static'))
    
    # Load configuration
    app.config.from_object(Config)
    
    # Register blueprints
    app.register_blueprint(main_routes.bp)
    app.register_blueprint(image_routes.bp)
    app.register_blueprint(video_routes.bp)
    
    # Preload models
    preload_models()
    
    return app 