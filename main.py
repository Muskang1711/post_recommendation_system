#!/usr/bin/env python3
"""
Main pipeline for Post Recommendation System
Complete end-to-end pipeline with logging, error handling, and configuration management
"""

import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml

# Import custom modules
from data_ingestion import DataIngestion
from validation.data_preprocessing import DataPreprocessor
from preprocessing.model import KNNRecommendationModel
from preprocessing.predict_model import RecommendationPredictor

def setup_logging(config: dict) -> None:
    """
    Setup logging configuration
    
    Args:
        config (dict): Configuration dictionary
    """
    log_config = config.get('logging', {})
    log_file = log_config.get('log_file', 'logs/recommendation_system.log')
    
    # Create logs directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging setup completed")

def load_config(config_path: str = "params.yaml") -> dict:
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
        return config
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        sys.exit(1)

def run_data_ingestion(config: dict) -> tuple:
    """
    Run data ingestion pipeline
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (post_data, user_data, view_data)
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("STARTING DATA INGESTION")
    logger.info("=" * 50)
    
    try:
        data_ingestion = DataIngestion(config_path="params.yaml")
        data_ingestion.create_directories()
        post_data, user_data, view_data = data_ingestion.load_all_data()
        
        logger.info("Data ingestion completed successfully")
        logger.info(f"Loaded datasets - Post: {post_data.shape}, User: {user_data.shape}, View: {view_data.shape}")
        
        return post_data, user_data, view_data
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        raise

def run_preprocessing(post_data, user_data, view_data, config: dict) -> tuple:
    """
    Run data preprocessing pipeline
    
    Args:
        post_data: Post dataframe
        user_data: User dataframe  
        view_data: View dataframe
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (processed_df, pivot_table, sparse_matrix)
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("STARTING DATA PREPROCESSING")
    logger.info("=" * 50)
    
    try:
        preprocessor = DataPreprocessor(config_path="params.yaml")
        processed_df, pivot_table, sparse_matrix = preprocessor.preprocess_data(
            view_data, post_data
        )
        
        logger.info("Data preprocessing completed successfully")
        logger.info(f"Processed data shape: {processed_df.shape}")
        logger.info(f"Pivot table shape: {pivot_table.shape}")
        logger.info(f"Unique titles: {processed_df['title'].nunique()}")
        logger.info(f"Unique users: {processed_df['user_id'].nunique()}")
        
        return processed_df, pivot_table, sparse_matrix
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        raise

def run_model_training(pivot_table, sparse_matrix, config: dict) -> KNNRecommendationModel:
    """
    Run model training pipeline
    
    Args:
        pivot_table: User-item pivot table
        sparse_matrix: Sparse matrix representation
        config (dict): Configuration dictionary
        
    Returns:
        KNNRecommendationModel: Trained model
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 50)
    
    try:
        model = KNNRecommendationModel(config_path="params.yaml")
        model.train(pivot_table, sparse_matrix)
        
        # Save the trained model
        model.save_model()
        
        logger.info("Model training completed successfully")
        logger.info(f"Model trained on {pivot_table.shape[0]} items and {pivot_table.shape[1]} users")
        
        return model
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

def run_evaluation(model: KNNRecommendationModel, config: dict) -> None:
    """
    Run model evaluation and generate sample predictions
    
    Args:
        model (KNNRecommendationModel): Trained model
        config (dict): Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("STARTING MODEL EVALUATION")
    logger.info("=" * 50)
    
    try:
        # Initialize predictor
        predictor = RecommendationPredictor(config_path="params.yaml")
        
        # Get model statistics
        stats = predictor.get_model_statistics()
        logger.info("Model Statistics:")
        logger.info(f"  Items: {stats['model_info']['n_items']}")
        logger.info(f"  Users: {stats['model_info']['n_users']}")
        logger.info(f"  Sparsity: {stats['model_info']['sparsity_percentage']:.2f}%")
        logger.info(f"  Mean Rating: {stats['rating_statistics']['mean_rating']:.2f}")
        
        # Get top-rated posts
        top_posts = predictor.get_top_posts(5)
        logger.info("\nTop 5 Posts:")
        for i, post in enumerate(top_posts['top_posts'], 1):
            logger.info(f"  {i}. {post['title']} (Avg: {post['avg_rating']:.2f})")
        
        # Generate sample recommendations
        logger.info("\nSample Recommendations:")
        for i in range(3):
            random_rec = predictor.get_random_recommendations(3)
            logger.info(f"\nRecommendations for: '{random_rec['query_title']}'")
            for j, rec in enumerate(random_rec['recommendations'], 1):
                logger.info(f"  {j}. {rec['title']} (sim: {rec['similarity_score']:.3f})")
        
        logger.info("Model evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise

def run_complete_pipeline(config_path: str = "params.yaml") -> None:
    """
    Run the complete recommendation system pipeline
    
    Args:
        config_path (str): Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    start_time = datetime.now()
    
    logger.info("*" * 70)
    logger.info("POST RECOMMENDATION SYSTEM PIPELINE STARTED")
    logger.info(f"Start Time: {start_time}")
    logger.info("*" * 70)
    
    try:
        # Step 1: Data Ingestion
        post_data, user_data, view_data = run_data_ingestion(config)
        
        # Step 2: Data Preprocessing
        processed_df, pivot_table, sparse_matrix = run_preprocessing(
            post_data, user_data, view_data, config
        )
        
        # Step 3: Model Training
        model = run_model_training(pivot_table, sparse_matrix, config)
        
        # Step 4: Model Evaluation
        run_evaluation(model, config)
        
        # Pipeline completed successfully
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        logger.info("*" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"End Time: {end_time}")
        logger.info(f"Total Runtime: {total_time:.2f} seconds")
        logger.info("*" * 70)
        
        print("\n" + "=" * 60)
        print("🎉 RECOMMENDATION SYSTEM PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Total Runtime: {total_time:.2f} seconds")
        print(f"📈 Items in Model: {pivot_table.shape[0]}")
        print(f"👥 Users in Model: {pivot_table.shape[1]}")
        print("📁 Model saved to: models/knn_model.pkl")
        print("📋 Logs saved to: logs/recommendation_system.log")
        print("=" * 60)
        
    except Exception as e:
        logger.error("Pipeline failed with error: %s", str(e))
        print(f"\n❌ Pipeline failed: {str(e)}")
        print("Check logs for detailed error information.")
        sys.exit(1)

def main():
    """
    Main function with command-line argument parsing
    """
    parser = argparse.ArgumentParser(
        description='Post Recommendation System Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --full-pipeline              # Run complete pipeline
  python main.py --train-only                 # Only train model (assumes data exists)
  python main.py --predict "Some Post Title"  # Get recommendations for a title
  python main.py --random                     # Get random recommendations
  python main.py --config custom_params.yaml # Use custom configuration
        """
    )
    
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run the complete pipeline (default)')
    parser.add_argument('--train-only', action='store_true',
                       help='Only train the model (assumes preprocessed data exists)')
    parser.add_argument('--predict', type=str, metavar='TITLE',
                       help='Get recommendations for a specific post title')
    parser.add_argument('--random', action='store_true',
                       help='Get random recommendations')
    parser.add_argument('--config', type=str, default='params.yaml',
                       help='Configuration file path (default: params.yaml)')
    parser.add_argument('--n-recs', type=int, default=5,
                       help='Number of recommendations (default: 5)')
    
    args = parser.parse_args()
    
    # If no specific action is chosen, run full pipeline
    if not any([args.train_only, args.predict, args.random]):
        args.full_pipeline = True
    
    if args.full_pipeline:
        # Run complete pipeline
        run_complete_pipeline(args.config)
        
    elif args.train_only:
        # Only train model
        config = load_config(args.config)
        setup_logging(config)
        
        try:
            # Load preprocessed data
            import pickle
            with open(config['output']['pivot_table_path'], 'rb') as f:
                pivot_table = pickle.load(f)
            
            from scipy.sparse import csr_matrix
            sparse_matrix = csr_matrix(pivot_table.values)
            
            # Train model
            model = run_model_training(pivot_table, sparse_matrix, config)
            print("✅ Model training completed successfully!")
            
        except FileNotFoundError:
            print("❌ Preprocessed data not found. Please run full pipeline first.")
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            
    elif args.predict or args.random:
        # Make predictions
        try:
            predictor = RecommendationPredictor(args.config)
            
            if args.predict:
                result = predictor.predict_single(args.predict, args.n_recs)
                print(f"\n🔍 Recommendations for: '{args.predict}'")
                print("-" * 60)
                
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    for i, rec in enumerate(result['recommendations'], 1):
                        print(f"{i:2d}. {rec['title']}")
                        print(f"    📊 Similarity: {rec['similarity_score']:.3f}")
                        
            elif args.random:
                result = predictor.get_random_recommendations(args.n_recs)
                print(f"\n🎲 Random recommendations for: '{result['query_title']}'")
                print("-" * 60)
                
                for i, rec in enumerate(result['recommendations'], 1):
                    print(f"{i:2d}. {rec['title']}")
                    print(f"    📊 Similarity: {rec['similarity_score']:.3f}")
                    
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            print("Make sure the model is trained first by running: python main.py --full-pipeline")

if __name__ == "__main__":
    main()