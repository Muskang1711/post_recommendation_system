import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import yaml
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logger = logging.getLogger(__name__)

class KNNRecommendationModel:
    """
    K-Nearest Neighbors based recommendation system for collaborative filtering
    """
    
    def __init__(self, config_path: str = "params.yaml"):
        """
        Initialize KNN Recommendation Model
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.model_params = self.config['model']
        self.model: Optional[NearestNeighbors] = None
        self.pivot_table: Optional[pd.DataFrame] = None
        self.sparse_matrix: Optional[csr_matrix] = None
        self.is_trained = False
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def initialize_model(self) -> NearestNeighbors:
        """
        Initialize KNN model with configured parameters
        
        Returns:
            NearestNeighbors: Initialized KNN model
        """
        logger.info("Initializing KNN model")
        
        model = NearestNeighbors(
            metric=self.model_params['metric'],
            algorithm=self.model_params['algorithm'],
            n_jobs=-1  # Use all available cores
        )
        
        logger.info(f"KNN model initialized with parameters: {self.model_params}")
        return model
    
    def train(self, pivot_table: pd.DataFrame, sparse_matrix: csr_matrix) -> None:
        """
        Train the KNN model on user-item matrix
        
        Args:
            pivot_table (pd.DataFrame): User-item pivot table
            sparse_matrix (csr_matrix): Sparse representation of pivot table
        """
        logger.info("Training KNN model")
        
        try:
            # Store data
            self.pivot_table = pivot_table
            self.sparse_matrix = sparse_matrix
            
            # Initialize and fit model
            self.model = self.initialize_model()
            self.model.fit(sparse_matrix)
            
            self.is_trained = True
            logger.info(f"Model training completed successfully on {sparse_matrix.shape[0]} items")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def get_recommendations(self, title: str, n_recommendations: int = None) -> List[Dict[str, Any]]:
        """
        Get recommendations for a given post title
        
        Args:
            title (str): Title of the post to base recommendations on
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            List[Dict[str, Any]]: List of recommendations with distances
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if n_recommendations is None:
            n_recommendations = self.model_params['n_neighbors']
        
        logger.info(f"Getting recommendations for title: '{title}'")
        
        try:
            # Check if title exists in pivot table
            if title not in self.pivot_table.index:
                available_titles = self.get_similar_titles(title)
                if available_titles:
                    logger.warning(f"Title '{title}' not found. Similar titles: {available_titles[:5]}")
                    raise ValueError(f"Title '{title}' not found in dataset. Similar titles: {available_titles[:5]}")
                else:
                    raise ValueError(f"Title '{title}' not found in dataset")
            
            # Get index of the title
            title_index = self.pivot_table.index.get_loc(title)
            
            # Get item vector
            title_vector = self.pivot_table.iloc[title_index, :].values.reshape(1, -1)
            
            # Find nearest neighbors
            distances, indices = self.model.kneighbors(
                title_vector, 
                n_neighbors=n_recommendations + 1  # +1 because it includes the input title
            )
            
            # Prepare recommendations (exclude the first one as it's the input title itself)
            recommendations = []
            for i in range(1, len(distances.flatten())):
                rec_index = indices.flatten()[i]
                recommendation = {
                    'title': self.pivot_table.index[rec_index],
                    'similarity_score': 1 - distances.flatten()[i],  # Convert distance to similarity
                    'distance': distances.flatten()[i],
                    'rank': i
                }
                recommendations.append(recommendation)
            
            logger.info(f"Generated {len(recommendations)} recommendations for '{title}'")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise
    
    def get_similar_titles(self, query_title: str, max_results: int = 10) -> List[str]:
        """
        Find titles similar to the query title (for handling typos/variations)
        
        Args:
            query_title (str): Query title
            max_results (int): Maximum number of similar titles to return
            
        Returns:
            List[str]: List of similar titles
        """
        if self.pivot_table is None:
            return []
        
        query_lower = query_title.lower()
        similar_titles = []
        
        for title in self.pivot_table.index:
            if query_lower in title.lower() or title.lower() in query_lower:
                similar_titles.append(title)
        
        return similar_titles[:max_results]
    
    def get_random_recommendation(self, n_recommendations: int = None) -> Dict[str, Any]:
        """
        Get recommendations for a random title (useful for exploration)
        
        Args:
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            Dict[str, Any]: Dictionary containing query title and recommendations
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if n_recommendations is None:
            n_recommendations = self.model_params['n_neighbors']
        
        # Select random title
        query_index = np.random.choice(self.pivot_table.shape[0])
        query_title = self.pivot_table.index[query_index]
        
        # Get recommendations
        recommendations = self.get_recommendations(query_title, n_recommendations)
        
        result = {
            'query_title': query_title,
            'recommendations': recommendations
        }
        
        logger.info(f"Generated random recommendations for: '{query_title}'")
        return result
    
    def calculate_item_similarity_matrix(self) -> pd.DataFrame:
        """
        Calculate item-item similarity matrix
        
        Returns:
            pd.DataFrame: Item similarity matrix
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating similarity matrix")
        
        logger.info("Calculating item similarity matrix")
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(self.sparse_matrix)
        
        # Convert to DataFrame for better readability
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=self.pivot_table.index,
            columns=self.pivot_table.index
        )
        
        logger.info(f"Item similarity matrix calculated with shape: {similarity_df.shape}")
        return similarity_df
    
    def get_top_rated_posts(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top-rated posts based on average ratings
        
        Args:
            top_n (int): Number of top posts to return
            
        Returns:
            pd.DataFrame: Top rated posts with statistics
        """
        if self.pivot_table is None:
            raise ValueError("Model must be trained before getting top rated posts")
        
        # Calculate statistics for each post
        post_stats = []
        for title in self.pivot_table.index:
            ratings = self.pivot_table.loc[title].values
            non_zero_ratings = ratings[ratings > 0]
            
            if len(non_zero_ratings) > 0:
                post_stats.append({
                    'title': title,
                    'avg_rating': np.mean(non_zero_ratings),
                    'total_ratings': len(non_zero_ratings),
                    'rating_std': np.std(non_zero_ratings)
                })
        
        # Convert to DataFrame and sort
        stats_df = pd.DataFrame(post_stats)
        top_posts = stats_df.nlargest(top_n, 'avg_rating')
        
        logger.info(f"Retrieved top {top_n} rated posts")
        return top_posts
    
    def save_model(self, model_path: str = None) -> None:
        """
        Save trained model and pivot table
        
        Args:
            model_path (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if model_path is None:
            model_path = self.config['output']['model_path']
        
        try:
            # Create directory if it doesn't exist
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model and data
            model_data = {
                'model': self.model,
                'pivot_table': self.pivot_table,
                'sparse_matrix': self.sparse_matrix,
                'config': self.config
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved successfully to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path: str = None) -> None:
        """
        Load trained model and pivot table
        
        Args:
            model_path (str): Path to load the model from
        """
        if model_path is None:
            model_path = self.config['output']['model_path']
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.pivot_table = model_data['pivot_table']
            self.sparse_matrix = model_data['sparse_matrix']
            self.is_trained = True
            
            logger.info(f"Model loaded successfully from {model_path}")
            
        except FileNotFoundError:
            logger.error(f"Model file not found at {model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

if __name__ == "__main__":
    from data_ingestion import DataIngestion
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    data_ingestion = DataIngestion()
    post_data, user_data, view_data = data_ingestion.load_all_data()
    
    preprocessor = DataPreprocessor()
    processed_df, pivot_table, sparse_matrix = preprocessor.preprocess_data(
        view_data, post_data
    )
    
    # Train model
    model = KNNRecommendationModel()
    model.train(pivot_table, sparse_matrix)
    
    # Test with random recommendations
    random_rec = model.get_random_recommendation()
    print(f"\nRecommendations for: {random_rec['query_title']}")
    for i, rec in enumerate(random_rec['recommendations'], 1):
        print(f"{i}. {rec['title']} (similarity: {rec['similarity_score']:.3f})")
    
    # Save model
    model.save_model()
    print("\nModel saved successfully!")