"""
Configuration file for CultivaSense Flask application
"""

import os
from pathlib import Path

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent

# Application Configuration
class Config:
    """Base configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    DEBUG = os.environ.get('FLASK_DEBUG', False)
    
    # Upload Configuration
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 5242880))  # 5MB default
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Model Paths
    CROP_MODEL_PATH = os.environ.get('CROP_MODEL_PATH', 'model.pkl')
    MINMAX_SCALER_PATH = os.environ.get('MINMAX_SCALER_PATH', 'minmaxscaler.pkl')
    DISEASE_MODEL_PATH = os.environ.get('DISEASE_MODEL_PATH', 'models/plant_disease_model.h5')
    PRICE_MODEL_PATH = os.environ.get('PRICE_MODEL_PATH', 'models/market_price_model.pkl')
    CSV_PATH = os.environ.get('CSV_PATH', 'Crop_recommendation.csv')
    
    # Data paths
    DISEASE_DATA_PATH = os.environ.get('DISEASE_DATA_PATH', 'Crop_Disease.csv')
    PRICE_DATA_PATH = os.environ.get('PRICE_DATA_PATH', 'market_price_data.csv')
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    CSRF_ENABLED = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    """Get configuration based on environment"""
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
