# Post Recommendation System

A comprehensive collaborative filtering-based recommendation system built with Python, featuring modular architecture, comprehensive logging, type hints, and configurable parameters.

## 🚀 Features

- **Collaborative Filtering**: KNN-based recommendation engine using cosine similarity
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Comprehensive Logging**: Detailed logging throughout the pipeline
- **Type Hints**: Full type annotation for better code maintainability
- **Configurable Parameters**: YAML-based configuration management
- **Batch Processing**: Support for both single and batch predictions
- **Model Persistence**: Save and load trained models
- **Export Functionality**: Export results to JSON or CSV formats
- **Command Line Interface**: Easy-to-use CLI for all operations

## 📁 Project Structure

```
post_recommendation_system/
│
├── params.yaml                 # Configuration file
├── requirements.txt            # Dependencies
├── README.md                  # This file
├── main.py                    # Main pipeline runner
│
├── data_ingestion.py          # Data loading and validation
├── data_preprocessing.py      # Data cleaning and preprocessing
├── model.py                   # KNN recommendation model
├── predict_model.py          # Prediction and inference
│
├── data/
│   ├── raw/                   # Raw data files
│   │   ├── post_data.csv
│   │   ├── user_data.csv
│   │   └── view_data.csv
│   └── processed/             # Processed data files
│
├── models/                    # Saved models
│   ├── knn_model.pkl
│   └── pivot_table.pkl
│
└── logs/                      # Log files
    └── recommendation_system.log
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd post_recommendation_system
```

2. **Create a virtual environment:**
```bash
python -m venv recommendation_env
source recommendation_env/bin/activate  # On Windows: recommendation_env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Prepare your data:**
   - Place your CSV files in the `data/raw/` directory:
     - `post_data.csv`: Contains post_id, title, category columns
     - `user_data.csv`: Contains user_id and other user information
     - `view_data.csv`: Contains user_id, post_id, time_stamp columns

## 🔧 Configuration

The system uses `params.yaml` for configuration. Key parameters:

```yaml
# Data paths
data:
  raw_data_path: "data/raw"
  processed_data_path: "data/processed"

# Model parameters
model:
  algorithm: 'brute'
  metric: 'cosine'
  n_neighbors: 6

# Preprocessing parameters
preprocessing:
  popularity_threshold: 13
  rating_range_min: 1
  rating_range_max: 6
```

## 🚀 Usage

### Full Pipeline (Recommended for first run)

```bash
python main.py --full-pipeline
```

This will:
1. Load and validate data
2. Preprocess and clean data
3. Train the KNN model
4. Evaluate the model
5. Save the trained model

### Individual Operations

**Get recommendations for a specific post:**
```bash
python main.py --predict "Your Post Title Here" --n-recs 5
```

**Get random recommendations:**
```bash
python main.py --random --n-recs 5
```

**Train model only (if data is already processed):**
```bash
python main.py --train-only
```

**Use custom configuration:**
```bash
python main.py --config custom_params.yaml --full-pipeline
```

### Using the Prediction Module Directly

```bash
# Get recommendations for a specific title
python predict_model.py --title "Your Post Title" --n-recommendations 5

# Get random recommendations
python predict_model.py --random

# Show model statistics
python predict_model.py --stats

# Get top-rated posts
python predict_model.py --top-posts

# Export results
python predict_model.py --title "Your Post Title" --output results.json --format json
```

## 🐍 Programmatic Usage

### Basic Usage

```python
from data_ingestion import DataIngestion
from data_preprocessing import DataPreprocessor
from model import KNNRecommendationModel
from predict_model import RecommendationPredictor

# Load data
data_ingestion = DataIngestion()
post_data, user_data, view_data = data_ingestion.load_all_data()

# Preprocess data
preprocessor = DataPreprocessor()
processed_df, pivot_table, sparse_matrix = preprocessor.preprocess_data(
    view_data, post_data
)

# Train model
model = KNNRecommendationModel()
model.train(pivot_table, sparse_matrix)

# Make predictions
predictor = RecommendationPredictor()
recommendations = predictor.predict_single("Your Post Title", n_recommendations=5)
print(recommendations)
```

### Advanced Usage

```python
# Get model statistics
predictor = RecommendationPredictor()
stats = predictor.get_model_statistics()
print(f"Model has {stats['model_info']['n_items']} items")

# Batch predictions
titles = ["Title 1", "Title 2", "Title 3"]
batch_results = predictor.predict_batch(titles, n_recommendations=3)

# Export results
predictor.export_recommendations(batch_results, "batch_results.csv", format="csv")
```

## 📊 Model Details

### Algorithm
- **Method**: K-Nearest Neighbors (KNN) with Collaborative Filtering
- **Similarity Metric**: Cosine Similarity
- **Matrix Factorization**: Sparse matrix representation for memory efficiency

### Data Processing Pipeline
1. **Data Validation**: Ensures required columns exist and data integrity
2. **Rating Generation**: Adds synthetic valuable ratings (1-5 scale)
3. **Data Merging**: Combines view and post data
4. **Popularity Filtering**: Filters posts based on minimum interaction threshold
5. **Matrix Creation**: Creates user-item interaction matrix
6. **Model Training**: Fits KNN model on the sparse matrix

### Performance Considerations
- Uses sparse matrices for memory efficiency
- Configurable popularity threshold to focus on relevant items
- Parallel processing support (`n_jobs=-1`)

## 📈 Evaluation Metrics

The system provides several evaluation insights:

- **Model Statistics**: Number of users, items, sparsity percentage
- **Rating Statistics**: Mean, median, standard deviation of ratings
- **Top-rated Posts**: Posts with highest average ratings
- **Similarity Scores**: Cosine similarity scores for recommendations

## 🔍 Logging

Comprehensive logging is implemented throughout the system:

- **File Logging**: All logs saved to `logs/recommendation_system.log`
- **Console Logging**: Real-time progress updates
- **Log Levels**: INFO, WARNING, ERROR with timestamps
- **Module-specific Logging**: Each module has dedicated logger

## 🐛 Troubleshooting

### Common Issues

1. **"Model not trained" error**:
   ```bash
   python main.py --full-pipeline
   ```

2. **"Title not found" error**:
   - Check if the title exists in your dataset
   - Use similar title search to find close matches

3. **Memory issues with large datasets**:
   - Increase popularity threshold in `params.yaml`
   - Consider data sampling for initial testing

4. **Import errors**:
   ```bash
   pip install -r requirements.txt
   ```

### Debug Mode

Enable detailed logging by modifying `params.yaml`:
```yaml
logging:
  level: "DEBUG"
```

## 📝 Data Format Requirements

### post_data.csv
```csv
post_id,title,category
1,"10 Funny ART Quotes","Art"
2,"Best Programming Tips","Tech"
```

### user_data.csv
```csv
user_id,city,age
1,"New York",25
2,"London",30
```

### view_data.csv
```csv
user_id,post_id,time_stamp
1,1,"2023-01-01 10:00:00"
2,1,"2023-01-01 11:00:00"
```

## 🔮 Future Enhancements

- [ ] Deep learning-based recommendations (Neural Collaborative Filtering)
- [ ] Real-time recommendation API
- [ ] A/B testing framework
- [ ] Content-based filtering integration
- [ ] Recommendation explanation features
- [ ] Web interface
- [ ] Docker containerization
- [ ] Cloud deployment scripts

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Review the logs in `logs/recommendation_system.log`
3. Open an issue on GitHub with detailed error information

## 🙏 Acknowledgments

- Scikit-learn for machine learning algorithms
- Pandas and NumPy for data manipulation
- SciPy for sparse matrix operations