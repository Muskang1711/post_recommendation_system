"""
Utility functions for the Post Recommendation System
"""

import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yaml

# Configure logging
logger = logging.getLogger(__name__)

class RecommendationUtils:
    """
    Utility class containing helper functions for the recommendation system
    """
    
    @staticmethod
    def validate_csv_files(file_paths: List[str]) -> Dict[str, bool]:
        """
        Validate if CSV files exist and are readable
        
        Args:
            file_paths (List[str]): List of file paths to validate
            
        Returns:
            Dict[str, bool]: Dictionary mapping file paths to validation status
        """
        validation_results = {}
        
        for file_path in file_paths:
            try:
                if Path(file_path).exists():
                    # Try to read the first few rows
                    pd.read_csv(file_path, nrows=1)
                    validation_results[file_path] = True
                    logger.info(f"✅ File validated: {file_path}")
                else:
                    validation_results[file_path] = False
                    logger.warning(f"❌ File not found: {file_path}")
            except Exception as e:
                validation_results[file_path] = False
                logger.error(f"❌ File validation failed for {file_path}: {str(e)}")
        
        return validation_results
    
    @staticmethod
    def calculate_data_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for a dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        stats = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'null_counts': df.isnull().sum().to_dict(),
            'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'unique_counts': {col: df[col].nunique() for col in df.columns}
        }
        
        # Add numeric statistics for numeric columns
        if stats['numeric_columns']:
            numeric_stats = df[stats['numeric_columns']].describe().to_dict()
            stats['numeric_statistics'] = numeric_stats
        
        logger.info(f"Calculated statistics for dataframe with shape {df.shape}")
        return stats
    
    @staticmethod
    def create_data_quality_report(dfs: Dict[str, pd.DataFrame], 
                                 output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create comprehensive data quality report
        
        Args:
            dfs (Dict[str, pd.DataFrame]): Dictionary of dataframes
            output_path (Optional[str]): Path to save the report
            
        Returns:
            Dict[str, Any]: Data quality report
        """
        logger.info("Creating data quality report")
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'datasets': {}
        }
        
        for name, df in dfs.items():
            dataset_report = {
                'name': name,
                'statistics': RecommendationUtils.calculate_data_statistics(df),
                'quality_issues': []
            }
            
            # Check for quality issues
            # High null percentage
            for col, null_pct in dataset_report['statistics']['null_percentages'].items():
                if null_pct > 50:
                    dataset_report['quality_issues'].append({
                        'type': 'high_null_percentage',
                        'column': col,
                        'percentage': null_pct
                    })
            
            # Duplicate rows
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                dataset_report['quality_issues'].append({
                    'type': 'duplicate_rows',
                    'count': int(duplicate_count),
                    'percentage': float(duplicate_count / len(df) * 100)
                })
            
            # Single-value columns
            for col in df.columns:
                if df[col].nunique() == 1:
                    dataset_report['quality_issues'].append({
                        'type': 'single_value_column',
                        'column': col
                    })
            
            report['datasets'][name] = dataset_report
        
        # Save report if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Data quality report saved to {output_path}")
        
        return report
    
    @staticmethod
    def visualize_rating_distribution(ratings: np.ndarray, 
                                    save_path: Optional[str] = None) -> None:
        """
        Create visualization of rating distribution
        
        Args:
            ratings (np.ndarray): Array of ratings
            save_path (Optional[str]): Path to save the plot
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Create subplot for histogram and box plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram
            ax1.hist(ratings, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title('Rating Distribution')
            ax1.set_xlabel('Rating Value')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(ratings, vert=True)
            ax2.set_title('Rating Box Plot')
            ax2.set_ylabel('Rating Value')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Rating distribution plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating rating distribution plot: {str(e)}")
    
    @staticmethod
    def create_similarity_heatmap(similarity_matrix: pd.DataFrame, 
                                top_n: int = 20,
                                save_path: Optional[str] = None) -> None:
        """
        Create heatmap of item similarity matrix
        
        Args:
            similarity_matrix (pd.DataFrame): Item similarity matrix
            top_n (int): Number of top items to include
            save_path (Optional[str]): Path to save the plot
        """
        try:
            # Select top N items (by average similarity)
            avg_similarity = similarity_matrix.mean(axis=1).sort_values(ascending=False)
            top_items = avg_similarity.head(top_n).index
            
            # Create subset matrix
            subset_matrix = similarity_matrix.loc[top_items, top_items]
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(subset_matrix, annot=False, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            
            plt.title(f'Item Similarity Heatmap (Top {top_n} Items)')
            plt.xlabel('Items')
            plt.ylabel('Items')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Similarity heatmap saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating similarity heatmap: {str(e)}")
    
    @staticmethod
    def format_recommendations_for_display(recommendations: List[Dict[str, Any]], 
                                         max_title_length: int = 50) -> str:
        """
        Format recommendations for nice console display
        
        Args:
            recommendations (List[Dict[str, Any]]): List of recommendations
            max_title_length (int): Maximum length for title display
            
        Returns:
            str: Formatted recommendations string
        """
        if not recommendations:
            return "No recommendations available."
        
        formatted_lines = []
        formatted_lines.append("📋 RECOMMENDATIONS")
        formatted_lines.append("=" * 60)
        
        for i, rec in enumerate(recommendations, 1):
            title = rec['title']
            if len(title) > max_title_length:
                title = title[:max_title_length - 3] + "..."
            
            similarity = rec.get('similarity_score', 0)
            
            formatted_lines.append(f"{i:2d}. {title}")
            formatted_lines.append(f"    📊 Similarity: {similarity:.3f}")
            
            if i < len(recommendations):
                formatted_lines.append("")
        
        return "\n".join(formatted_lines)
    
    @staticmethod
    def benchmark_model_performance(model, pivot_table, n_tests: int = 100) -> Dict[str, float]:
        """
        Benchmark model inference performance
        
        Args:
            model: Trained recommendation model
            pivot_table (pd.DataFrame): User-item pivot table
            n_tests (int): Number of test runs
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        logger.info(f"Benchmarking model performance with {n_tests} tests")
        
        inference_times = []
        
        for _ in range(n_tests):
            # Select random title
            random_title = np.random.choice(pivot_table.index)
            
            # Measure inference time
            start_time = datetime.now()
            try:
                model.get_recommendations(random_title, 5)
                end_time = datetime.now()
                inference_time = (end_time - start_time).total_seconds()
                inference_times.append(inference_time)
            except Exception:
                continue
        
        if not inference_times:
            return {'error': 'No successful inference runs'}
        
        metrics = {
            'mean_inference_time': np.mean(inference_times),
            'median_inference_time': np.median(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'total_tests': len(inference_times),
            'success_rate': len(inference_times) / n_tests
        }
        
        logger.info(f"Performance benchmark completed: {metrics['mean_inference_time']:.4f}s mean")
        return metrics
    
    @staticmethod
    def export_model_summary(model, config: dict, output_path: str) -> None:
        """
        Export comprehensive model summary
        
        Args:
            model: Trained recommendation model
            config (dict): Configuration dictionary
            output_path (str): Path to save the summary
        """
        try:
            summary = {
                'model_info': {
                    'type': 'KNN_Collaborative_Filtering',
                    'algorithm': config['model']['algorithm'],
                    'metric': config['model']['metric'],
                    'n_neighbors': config['model']['n_neighbors'],
                    'created_at': datetime.now().isoformat()
                },
                'data_info': {
                    'n_items': model.pivot_table.shape[0] if model.pivot_table is not None else 0,
                    'n_users': model.pivot_table.shape[1] if model.pivot_table is not None else 0,
                    'sparsity': None,
                    'total_interactions': None
                },
                'configuration': config,
                'file_paths': {
                    'model_file': config['output']['model_path'],
                    'pivot_table_file': config['output']['pivot_table_path'],
                    'processed_data_file': config['output']['processed_data_path']
                }
            }
            
            # Calculate sparsity if model is trained
            if model.sparse_matrix is not None:
                total_possible = model.sparse_matrix.shape[0] * model.sparse_matrix.shape[1]
                actual_interactions = model.sparse_matrix.nnz
                summary['data_info']['sparsity'] = (1 - actual_interactions / total_possible) * 100
                summary['data_info']['total_interactions'] = actual_interactions
            
            # Save summary
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Model summary exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting model summary: {str(e)}")
            raise

# Convenience functions
def load_config(config_path: str = "params.yaml") -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_directories(config: dict) -> None:
    """Create all necessary directories"""
    directories = [
        config['data']['raw_data_path'],
        config['data']['processed_data_path'], 
        Path(config['output']['model_path']).parent,
        Path(config['logging']['log_file']).parent
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def print_banner(text: str, char: str = "=", width: int = 60) -> None:
    """Print a formatted banner"""
    print(char * width)
    print(f"{text:^{width}}")
    print(char * width)

if __name__ == "__main__":
    # Example usage
    utils = RecommendationUtils()
    
    # Validate CSV files
    files_to_check = [
        "data/raw/post_data.csv",
        "data/raw/user_data.csv", 
        "data/raw/view_data.csv"
    ]
    
    validation_results = utils.validate_csv_files(files_to_check)
    print("File validation results:")
    for file_path, is_valid in validation_results.items():
        status = "✅" if is_valid else "❌"
        print(f"{status} {file_path}")
    
    # Print banner example
    print_banner("POST RECOMMENDATION SYSTEM")
    print_banner("Data Processing Complete", char="-", width=50)