import pandas as pd
import numpy as np
import logging
import os
from typing import Tuple, Optional
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Class for handling data ingestion and initial data loading
    """
    
    def __init__(self, config_path: str = "params.yaml"):
        """
        Initialize DataIngestion with configuration
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.data_path = Path(self.config['data']['raw_data_path'])
        
    def load_config(self, config_path: str) -> dict:
        """
        Load configuration from YAML file
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded successfully from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def load_csv_data(self, filename: str) -> pd.DataFrame:
        """
        Load CSV data with error handling
        
        Args:
            filename (str): Name of the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            file_path = self.data_path / filename
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {filename} with shape {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File {filename} not found at {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all required datasets
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            post_data, user_data, view_data
        """
        logger.info("Starting data ingestion process")
        
        # Load all datasets
        post_data = self.load_csv_data(self.config['data']['post_data_file'])
        user_data = self.load_csv_data(self.config['data']['user_data_file'])
        view_data = self.load_csv_data(self.config['data']['view_data_file'])
        
        # Validate data
        self.validate_data(post_data, user_data, view_data)
        
        logger.info("Data ingestion completed successfully")
        return post_data, user_data, view_data
    
    def validate_data(self, post_data: pd.DataFrame, 
                     user_data: pd.DataFrame, 
                     view_data: pd.DataFrame) -> None:
        """
        Validate loaded data for basic requirements
        
        Args:
            post_data (pd.DataFrame): Post data
            user_data (pd.DataFrame): User data
            view_data (pd.DataFrame): View data
        """
        # Check if dataframes are empty
        if post_data.empty or user_data.empty or view_data.empty:
            raise ValueError("One or more datasets are empty")
        
        # Check for required columns
        required_post_cols = ['post_id', 'title']
        required_user_cols = ['user_id']
        required_view_cols = ['user_id', 'post_id']
        
        for col in required_post_cols:
            if col not in post_data.columns:
                raise ValueError(f"Required column '{col}' missing in post_data")
        
        for col in required_user_cols:
            if col not in user_data.columns:
                raise ValueError(f"Required column '{col}' missing in user_data")
        
        for col in required_view_cols:
            if col not in view_data.columns:
                raise ValueError(f"Required column '{col}' missing in view_data")
        
        logger.info("Data validation completed successfully")
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        directories = [
            self.config['data']['processed_data_path'],
            'models',
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created/verified: {directory}")

if __name__ == "__main__":
    # Example usage
    data_ingestion = DataIngestion()
    data_ingestion.create_directories()
    post_data, user_data, view_data = data_ingestion.load_all_data()
    
    print("Data shapes:")
    print(f"Post data: {post_data.shape}")
    print(f"User data: {user_data.shape}")
    print(f"View data: {view_data.shape}")
    
    print("\nData preview:")
    print("Post data head:")
    print(post_data.head())
    print("\nUser data head:")
    print(user_data.head())
    print("\nView data head:")
    print(view_data.head())