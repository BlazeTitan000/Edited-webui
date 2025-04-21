import os
from app import create_app

# Ensure upload directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Create and run the application
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    app.run(host=host, port=port, debug=False) 