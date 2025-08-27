import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Optional
from pathlib import Path
import yaml
from scipy.sparse import csr_matrix
import pickle

# Configure logging
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Class for handling data preprocessing and feature engineering
    """
    
    def __init__(self, config_path: str = "params.yaml"):
        """
        Initialize DataPreprocessor with configuration
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.popularity_threshold = self.config['preprocessing']['popularity_threshold']
        self.rating_range = (
            self.config['preprocessing']['rating_range_min'],
            self.config['preprocessing']['rating_range_max']
        )
        self.drop_columns = self.config['preprocessing']['drop_columns']
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def add_valuable_ratings(self, view_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add random valuable ratings to view data
        
        Args:
            view_data (pd.DataFrame): Original view data
            
        Returns:
            pd.DataFrame: View data with valuable ratings
        """
        logger.info("Adding valuable ratings to view data")
        
        # Create a copy to avoid modifying original data
        dataframe = view_data.copy()
        
        # Add random ratings
        np.random.seed(42)  # For reproducibility
        dataframe["Valuable"] = np.random.randint(
            self.rating_range[0], 
            self.rating_range[1], 
            len(dataframe)
        )
        
        logger.info(f"Added valuable ratings with range {self.rating_range}")
        return dataframe
    
    def merge_datasets(self, view_data: pd.DataFrame, 
                      post_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge view data with post data
        
        Args:
            view_data (pd.DataFrame): View data with ratings
            post_data (pd.DataFrame): Post data
            
        Returns:
            pd.DataFrame: Merged dataframe
        """
        logger.info("Merging view data with post data")
        
        try:
            df = pd.merge(view_data, post_data, on='post_id', how='inner')
            logger.info(f"Successfully merged datasets. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by dropping irrelevant columns and handling missing values
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Cleaning data")
        
        # Drop specified columns if they exist
        columns_to_drop = [col for col in self.drop_columns if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns_to_drop, axis=1)
            logger.info(f"Dropped columns: {columns_to_drop}")
        
        # Remove rows with null titles
        initial_shape = df.shape
        df = df.dropna(axis=0, subset=['title'])
        logger.info(f"Removed {initial_shape[0] - df.shape[0]} rows with null titles")
        
        return df
    
    def calculate_post_popularity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate total rating count for each post
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with popularity metrics
        """
        logger.info("Calculating post popularity metrics")
        
        # Calculate total valuable count for each title
        post_rating_count = (df
                           .groupby(by=['title'])['Valuable']
                           .count()
                           .reset_index()
                           .rename(columns={'Valuable': 'totalValuableCount'})
                           [['title', 'totalValuableCount']])
        
        # Merge back with original data
        df_with_popularity = df.merge(post_rating_count, on='title', how='left')
        
        logger.info(f"Calculated popularity for {len(post_rating_count)} unique posts")
        logger.info(f"Popularity statistics:")
        logger.info(f"Mean: {post_rating_count['totalValuableCount'].mean():.2f}")
        logger.info(f"Median: {post_rating_count['totalValuableCount'].median():.2f}")
        logger.info(f"Max: {post_rating_count['totalValuableCount'].max()}")
        
        return df_with_popularity
    
    def filter_popular_posts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter posts based on popularity threshold
        
        Args:
            df (pd.DataFrame): Dataframe with popularity metrics
            
        Returns:
            pd.DataFrame: Filtered dataframe with popular posts only
        """
        logger.info(f"Filtering posts with popularity threshold: {self.popularity_threshold}")
        
        initial_shape = df.shape
        popular_posts = df.query('totalValuableCount >= @self.popularity_threshold')
        
        logger.info(f"Filtered from {initial_shape[0]} to {popular_posts.shape[0]} rows")
        logger.info(f"Unique posts after filtering: {popular_posts['title'].nunique()}")
        
        return popular_posts
    
    def create_user_item_matrix(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, csr_matrix]:
        """
        Create user-item matrix for collaborative filtering
        
        Args:
            df (pd.DataFrame): Processed dataframe
            
        Returns:
            Tuple[pd.DataFrame, csr_matrix]: Pivot table and sparse matrix
        """
        logger.info("Creating user-item matrix")
        
        # Remove duplicates
        df_clean = df.drop_duplicates(['user_id', 'title'])
        logger.info(f"Removed duplicates. Shape: {df_clean.shape}")
        
        # Create pivot table
        pivot_table = df_clean.pivot(
            index='title', 
            columns='user_id', 
            values='Valuable'
        ).fillna(0)
        
        # Create sparse matrix
        sparse_matrix = csr_matrix(pivot_table.values)
        
        logger.info(f"Created user-item matrix with shape: {pivot_table.shape}")
        logger.info(f"Matrix sparsity: {(1 - sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1])) * 100:.2f}%")
        
        return pivot_table, sparse_matrix
    
    def save_processed_data(self, df: pd.DataFrame, 
                           pivot_table: pd.DataFrame) -> None:
        """
        Save processed data to files
        
        Args:
            df (pd.DataFrame): Processed dataframe
            pivot_table (pd.DataFrame): User-item pivot table
        """
        try:
            # Save processed dataframe
            processed_path = Path(self.config['output']['processed_data_path'])
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(processed_path, index=False)
            
            # Save pivot table
            pivot_path = Path(self.config['output']['pivot_table_path'])
            pivot_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pivot_path, 'wb') as f:
                pickle.dump(pivot_table, f)
            
            logger.info("Processed data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def preprocess_data(self, view_data: pd.DataFrame, 
                       post_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, csr_matrix]:
        """
        Complete preprocessing pipeline
        
        Args:
            view_data (pd.DataFrame): Raw view data
            post_data (pd.DataFrame): Raw post data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, csr_matrix]: 
            Processed dataframe, pivot table, and sparse matrix
        """
        logger.info("Starting complete preprocessing pipeline")
        
        # Step 1: Add valuable ratings
        view_with_ratings = self.add_valuable_ratings(view_data)
        
        # Step 2: Merge datasets
        merged_data = self.merge_datasets(view_with_ratings, post_data)
        
        # Step 3: Clean data
        cleaned_data = self.clean_data(merged_data)
        
        # Step 4: Calculate popularity
        data_with_popularity = self.calculate_post_popularity(cleaned_data)
        
        # Step 5: Filter popular posts
        popular_posts = self.filter_popular_posts(data_with_popularity)
        
        # Step 6: Create user-item matrix
        pivot_table, sparse_matrix = self.create_user_item_matrix(popular_posts)
        
        # Step 7: Save processed data
        self.save_processed_data(popular_posts, pivot_table)
        
        logger.info("Preprocessing pipeline completed successfully")
        return popular_posts, pivot_table, sparse_matrix

if __name__ == "__main__":
    from data_ingestion import DataIngestion
    
    # Load data
    data_ingestion = DataIngestion()
    post_data, user_data, view_data = data_ingestion.load_all_data()
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    processed_df, pivot_table, sparse_matrix = preprocessor.preprocess_data(
        view_data, post_data
    )
    
    print(f"Processed data shape: {processed_df.shape}")
    print(f"Pivot table shape: {pivot_table.shape}")
    print(f"Sparse matrix shape: {sparse_matrix.shape}")
    print(f"Unique titles: {processed_df['title'].nunique()}")
    print(f"Unique users: {processed_df['user_id'].nunique()}")