import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import yaml
import json
from datetime import datetime
from model import KNNRecommendationModel

# Configure logging
logger = logging.getLogger(__name__)

class RecommendationPredictor:
    """
    Class for making predictions and serving recommendations
    """
    
    def __init__(self, config_path: str = "params.yaml", model_path: str = None):
        """
        Initialize RecommendationPredictor
        
        Args:
            config_path (str): Path to configuration file
            model_path (str): Path to trained model file
        """
        self.config = self.load_config(config_path)
        self.model = KNNRecommendationModel(config_path)
        
        # Load trained model
        if model_path:
            self.model.load_model(model_path)
        else:
            try:
                self.model.load_model()
                logger.info("Model loaded successfully")
            except FileNotFoundError:
                logger.warning("No pre-trained model found. Please train the model first.")
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def predict_single(self, title: str, n_recommendations: int = 5) -> Dict[str, Any]:
        """
        Get recommendations for a single post title
        
        Args:
            title (str): Post title to get recommendations for
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            Dict[str, Any]: Prediction results with metadata
        """
        logger.info(f"Getting recommendations for: '{title}'")
        
        try:
            start_time = datetime.now()
            
            # Get recommendations from model
            recommendations = self.model.get_recommendations(title, n_recommendations)
            
            end_time = datetime.now()
            inference_time = (end_time - start_time).total_seconds()
            
            # Format response
            result = {
                'query_title': title,
                'recommendations': recommendations,
                'metadata': {
                    'inference_time_seconds': inference_time,
                    'timestamp': end_time.isoformat(),
                    'model_type': 'KNN_Collaborative_Filtering',
                    'n_recommendations': len(recommendations)
                }
            }
            
            logger.info(f"Generated {len(recommendations)} recommendations in {inference_time:.3f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {
                'query_title': title,
                'recommendations': [],
                'error': str(e),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model_type': 'KNN_Collaborative_Filtering',
                    'n_recommendations': 0
                }
            }
    
    def predict_batch(self, titles: List[str], n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Get recommendations for multiple post titles
        
        Args:
            titles (List[str]): List of post titles
            n_recommendations (int): Number of recommendations per title
            
        Returns:
            List[Dict[str, Any]]: List of prediction results
        """
        logger.info(f"Getting batch recommendations for {len(titles)} titles")
        
        results = []
        for title in titles:
            result = self.predict_single(title, n_recommendations)
            results.append(result)
        
        logger.info(f"Completed batch prediction for {len(titles)} titles")
        return results
    
    def get_random_recommendations(self, n_recommendations: int = 5) -> Dict[str, Any]:
        """
        Get recommendations for a randomly selected post
        
        Args:
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            Dict[str, Any]: Random recommendation results
        """
        logger.info("Getting random recommendations")
        
        try:
            # Get random recommendation from model
            random_result = self.model.get_random_recommendation(n_recommendations)
            
            # Format response
            result = {
                'query_title': random_result['query_title'],
                'recommendations': random_result['recommendations'],
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model_type': 'KNN_Collaborative_Filtering',
                    'recommendation_type': 'random',
                    'n_recommendations': len(random_result['recommendations'])
                }
            }
            
            logger.info(f"Generated random recommendations for: '{random_result['query_title']}'")
            return result
            
        except Exception as e:
            logger.error(f"Error in random recommendation: {str(e)}")
            raise
    
    def get_top_posts(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Get top-rated posts from the dataset
        
        Args:
            top_n (int): Number of top posts to return
            
        Returns:
            Dict[str, Any]: Top posts with ratings information
        """
        logger.info(f"Getting top {top_n} posts")
        
        try:
            top_posts_df = self.model.get_top_rated_posts(top_n)
            
            # Convert to dictionary format
            top_posts = top_posts_df.to_dict('records')
            
            result = {
                'top_posts': top_posts,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model_type': 'KNN_Collaborative_Filtering',
                    'query_type': 'top_rated',
                    'n_posts': len(top_posts)
                }
            }
            
            logger.info(f"Retrieved {len(top_posts)} top-rated posts")
            return result
            
        except Exception as e:
            logger.error(f"Error getting top posts: {str(e)}")
            raise
    
    def search_similar_titles(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Search for titles similar to the query
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            Dict[str, Any]: Similar titles with metadata
        """
        logger.info(f"Searching for titles similar to: '{query}'")
        
        try:
            similar_titles = self.model.get_similar_titles(query, max_results)
            
            result = {
                'query': query,
                'similar_titles': similar_titles,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model_type': 'KNN_Collaborative_Filtering',
                    'query_type': 'similar_search',
                    'n_results': len(similar_titles)
                }
            }
            
            logger.info(f"Found {len(similar_titles)} similar titles")
            return result
            
        except Exception as e:
            logger.error(f"Error in title search: {str(e)}")
            raise
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the trained model
        
        Returns:
            Dict[str, Any]: Model statistics and information
        """
        logger.info("Getting model statistics")
        
        if not self.model.is_trained:
            return {
                'error': 'Model not trained',
                'metadata': {
                    'timestamp': datetime.now().isoformat()
                }
            }
        
        try:
            # Get basic statistics
            n_items = self.model.pivot_table.shape[0]
            n_users = self.model.pivot_table.shape[1]
            
            # Calculate sparsity
            total_interactions = self.model.sparse_matrix.nnz
            possible_interactions = n_items * n_users
            sparsity = (1 - total_interactions / possible_interactions) * 100
            
            # Get rating statistics
            ratings = self.model.pivot_table.values[self.model.pivot_table.values > 0]
            
            statistics = {
                'model_info': {
                    'n_items': n_items,
                    'n_users': n_users,
                    'total_interactions': total_interactions,
                    'sparsity_percentage': round(sparsity, 2),
                    'model_algorithm': self.model.model_params['algorithm'],
                    'similarity_metric': self.model.model_params['metric']
                },
                'rating_statistics': {
                    'mean_rating': float(np.mean(ratings)),
                    'median_rating': float(np.median(ratings)),
                    'min_rating': float(np.min(ratings)),
                    'max_rating': float(np.max(ratings)),
                    'std_rating': float(np.std(ratings))
                },
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model_type': 'KNN_Collaborative_Filtering'
                }
            }
            
            logger.info("Model statistics calculated successfully")
            return statistics
            
        except Exception as e:
            logger.error(f"Error calculating model statistics: {str(e)}")
            raise
    
    def export_recommendations(self, results: Union[Dict, List[Dict]], 
                             output_path: str, format: str = 'json') -> None:
        """
        Export recommendation results to file
        
        Args:
            results (Union[Dict, List[Dict]]): Recommendation results
            output_path (str): Path to save the results
            format (str): Export format ('json' or 'csv')
        """
        logger.info(f"Exporting recommendations to {output_path}")
        
        try:
            # Create directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                    
            elif format.lower() == 'csv':
                # Convert to DataFrame for CSV export
                if isinstance(results, list):
                    # Batch results
                    flattened_data = []
                    for result in results:
                        query_title = result['query_title']
                        for rec in result.get('recommendations', []):
                            flattened_data.append({
                                'query_title': query_title,
                                'recommended_title': rec['title'],
                                'similarity_score': rec['similarity_score'],
                                'rank': rec['rank']
                            })
                else:
                    # Single result
                    flattened_data = []
                    query_title = results['query_title']
                    for rec in results.get('recommendations', []):
                        flattened_data.append({
                            'query_title': query_title,
                            'recommended_title': rec['title'],
                            'similarity_score': rec['similarity_score'],
                            'rank': rec['rank']
                        })
                
                df = pd.DataFrame(flattened_data)
                df.to_csv(output_path, index=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Results exported successfully to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            raise
    
    def validate_input(self, title: str) -> bool:
        """
        Validate input title
        
        Args:
            title (str): Title to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(title, str) or len(title.strip()) == 0:
            return False
        return True

def main():
    """
    Main function for command-line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Post Recommendation System Predictor')
    parser.add_argument('--title', type=str, help='Post title to get recommendations for')
    parser.add_argument('--n-recommendations', type=int, default=5, 
                       help='Number of recommendations (default: 5)')
    parser.add_argument('--random', action='store_true', 
                       help='Get random recommendations')
    parser.add_argument('--top-posts', action='store_true', 
                       help='Get top-rated posts')
    parser.add_argument('--stats', action='store_true', 
                       help='Show model statistics')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--format', type=str, default='json', 
                       choices=['json', 'csv'], help='Output format')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = RecommendationPredictor()
    
    result = None
    
    if args.title:
        # Get recommendations for specific title
        if not predictor.validate_input(args.title):
            print("Error: Invalid title provided")
            return
        
        result = predictor.predict_single(args.title, args.n_recommendations)
        print(f"\nRecommendations for: '{result['query_title']}'")
        print("-" * 60)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"{i:2d}. {rec['title']}")
                print(f"     Similarity: {rec['similarity_score']:.3f}")
                print()
    
    elif args.random:
        # Get random recommendations
        result = predictor.get_random_recommendations(args.n_recommendations)
        print(f"\nRandom recommendations for: '{result['query_title']}'")
        print("-" * 60)
        
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i:2d}. {rec['title']}")
            print(f"     Similarity: {rec['similarity_score']:.3f}")
            print()
    
    elif args.top_posts:
        # Get top-rated posts
        result = predictor.get_top_posts(args.n_recommendations)
        print(f"\nTop {args.n_recommendations} Posts:")
        print("-" * 60)
        
        for i, post in enumerate(result['top_posts'], 1):
            print(f"{i:2d}. {post['title']}")
            print(f"     Avg Rating: {post['avg_rating']:.2f} ({post['total_ratings']} ratings)")
            print()
    
    elif args.stats:
        # Show model statistics
        result = predictor.get_model_statistics()
        print("\nModel Statistics:")
        print("-" * 40)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            model_info = result['model_info']
            rating_stats = result['rating_statistics']
            
            print(f"Items: {model_info['n_items']}")
            print(f"Users: {model_info['n_users']}")
            print(f"Total Interactions: {model_info['total_interactions']}")
            print(f"Sparsity: {model_info['sparsity_percentage']:.2f}%")
            print(f"Algorithm: {model_info['model_algorithm']}")
            print(f"Metric: {model_info['similarity_metric']}")
            print()
            print("Rating Statistics:")
            print(f"  Mean: {rating_stats['mean_rating']:.2f}")
            print(f"  Median: {rating_stats['median_rating']:.2f}")
            print(f"  Range: {rating_stats['min_rating']:.1f} - {rating_stats['max_rating']:.1f}")
            print(f"  Std Dev: {rating_stats['std_rating']:.2f}")
    
    else:
        print("Please specify --title, --random, --top-posts, or --stats")
        return
    
    # Export results if output path is provided
    if args.output and result:
        predictor.export_recommendations(result, args.output, args.format)
        print(f"\nResults exported to: {args.output}")

if __name__ == "__main__":
    main()