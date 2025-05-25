# PyTorch Frame Examples

A comprehensive demonstration of PyTorch Frame capabilities for tabular deep learning, featuring both simple demonstrations and enhanced examples with advanced analysis.

## 🚀 Features

### **Simple Demonstrations (`pytorch_frame_simple_demos.py`)**
- **Modular Architecture**: Clean, reusable functions for different ML tasks
- **Binary Classification**: Senior citizen prediction (age >= 50)
- **Regression**: Age prediction with noise
- **Multi-class Classification**: Age group classification (4 categories)
- **Consistent API**: Standardized workflow across all task types

### **Enhanced Examples (`pytorch_frame_enhanced_examples.py`)**
- **Binary Classification**: Age-based senior classification, high earner prediction, education level prediction
- **Regression**: Education years prediction, age prediction, capital score prediction  
- **Multi-class Classification**: Work sector classification, age group classification, education tier classification
- **Comprehensive PCA Analysis**: 6-phase analysis including variance analysis, feature importance, target separability, visualization, feature clustering, and anomaly detection
- **Model Architectures**: Specialized models optimized for each task type
- **Performance Evaluation**: Task-specific metrics (accuracy, AUC, F1, MSE, MAE, RMSE)

### **Code Quality Features**
- **Modular Design**: Well-separated functions for different tasks and reusable components
- **Configuration Management**: Centralized configuration with `ExperimentConfig` and `PCAConfig`
- **Input Validation**: Comprehensive validation to prevent silent failures
- **Error Handling**: Proper logging and exception handling throughout

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA 12.8 (for GPU support, optional)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
- **PyTorch**: Deep learning framework with CUDA support
- **PyTorch Frame**: Tabular deep learning library
- **scikit-learn**: PCA, clustering, and metrics
- **pandas/polars**: Data manipulation
- **matplotlib**: Visualization
- **numpy**: Numerical computing

## 🎯 Usage

### Simple Demonstrations
Perfect for learning PyTorch Frame basics:
```python
# Run simple demonstrations
python pytorch_frame_simple_demos.py

# The script will automatically:
# 1. Run binary classification demo (senior citizen prediction)
# 2. Run regression demo (age prediction with noise)
# 3. Run multi-class classification demo (age groups)
# 4. Display training progress and final results
```

### Enhanced Examples
For comprehensive analysis and advanced features:
```python
# Run enhanced examples with PCA analysis (CPU recommended for PCA)
python pytorch_frame_enhanced_examples.py

# The script will automatically:
# 1. Load and prepare the Adult dataset
# 2. Create enhanced targets for different ML tasks
# 3. Perform comprehensive PCA analysis (CPU only)
# 4. Train and evaluate models for all task types
# 5. Display performance metrics and timing
```

### GPU Support
Both scripts automatically detect and use GPU if available:
```python
# GPU will be used automatically if CUDA is available
# Performance comparison between CPU and GPU will be shown
```

## 🏗️ Architecture

### Simple Demonstrations Structure

The `pytorch_frame_simple_demos.py` follows a clean, modular architecture:

#### **Core Components**
- `SimpleModel`: Unified model class for all task types
- `create_simple_tensor_frame()`: TensorFrame creation utility

#### **Data Preparation Functions**
- `load_and_prepare_data()`: Dataset loading and basic setup
- `split_data()`: Train/validation data splitting
- `create_data_loaders()`: PyTorch data loader creation
- `prepare_binary_data()`: Binary classification target preparation
- `prepare_regression_data()`: Regression target preparation  
- `prepare_multiclass_data()`: Multi-class target preparation

#### **Training Functions**
- `setup_model_and_optimizer()`: Model and optimizer initialization
- `train_epoch_binary()`, `validate_epoch_binary()`: Binary classification training/validation
- `train_epoch_regression()`, `validate_epoch_regression()`: Regression training/validation
- `train_epoch_multiclass()`, `validate_epoch_multiclass()`: Multi-class training/validation

#### **Demo Functions**
- `demo_binary_classification()`: Complete binary classification workflow
- `demo_regression()`: Complete regression workflow
- `demo_multiclass_classification()`: Complete multi-class workflow

### Enhanced Examples Structure

#### **Model Types**
1. **BinaryClassificationModel**: Optimized for binary tasks with dropout and specialized architecture
2. **RegressionModel**: Multi-layer architecture optimized for continuous predictions
3. **MultiClassModel**: Attention-like mechanism for multi-class classification

#### **Data Pipeline**
1. **Data Loading**: Yandex Adult dataset
2. **Target Creation**: 9 diverse targets across different task types
3. **Feature Engineering**: Numerical and categorical feature handling
4. **Data Splitting**: Configurable train/validation/test splits
5. **TensorFrame Creation**: PyTorch Frame's optimized data structure

#### **PCA Analysis Pipeline**
1. **Feature Preparation**: One-hot encoding and standardization
2. **Variance Analysis**: Component selection for different variance thresholds
3. **Feature Importance**: Analysis of feature contributions to principal components
4. **Target Separability**: Silhouette analysis for classification tasks
5. **Visualization**: 2D scatter plots colored by targets
6. **Feature Clustering**: K-means clustering of features based on PCA loadings
7. **Anomaly Detection**: Reconstruction error-based anomaly detection

## 📊 Output Examples

### Simple Demonstrations Output
```
🚀 PYTORCH FRAME SIMPLE DEMONSTRATIONS
Showcasing different task types with quick examples

============================================================
BINARY CLASSIFICATION DEMO
============================================================
Training binary classifier (senior citizen prediction)...
Epoch 1/5 - Train Loss: 0.6234, Train Acc: 65.42%, Val Acc: 67.83%
Epoch 2/5 - Train Loss: 0.5891, Train Acc: 69.17%, Val Acc: 71.50%
...
✅ Binary Classification Demo Complete!
   Final Validation Accuracy: 74.33%

============================================================
REGRESSION DEMO
============================================================
Training regression model (age prediction)...
Epoch 1/5 - Train MSE: 156.7834, Val MSE: 159.2341, Val RMSE: 12.6189
...
✅ Regression Demo Complete!
   Final Validation RMSE: 10.8456

============================================================
MULTI-CLASS CLASSIFICATION DEMO
============================================================
Training multi-class classifier (4 age groups)...
Epoch 1/5 - Train Loss: 1.2456, Train Acc: 45.67%, Val Acc: 47.83%
...
✅ Multi-Class Classification Demo Complete!
   Final Validation Accuracy: 52.17%

============================================================
🎯 ALL DEMOS COMPLETED SUCCESSFULLY!
============================================================
✅ Binary Classification: Senior citizen prediction
✅ Regression: Age prediction with noise
✅ Multi-Class: Age group classification (4 classes)

🚀 PyTorch Frame handles all tabular learning scenarios!
```

### Enhanced Examples Output
```
================================================================================
PYTORCH FRAME ENHANCED TRAINING EXAMPLES (Device: cpu)
================================================================================

1. Loading and preparing enhanced dataset...
   Dataset size: 5000 samples

=== PERFORMING PCA ANALYSIS ===
================================================================================
COMPREHENSIVE PCA ANALYSIS FOR DATASET UNDERSTANDING
================================================================================

1. Preparing features for PCA analysis...
   Total features: 45
   Numerical features: 6
   Categorical features (OHE): 39

2. Performing PCA variance analysis...
   Variance explained by component count:
     80% variance: 12 components
     90% variance: 18 components
     95% variance: 25 components
     99% variance: 35 components
```

## 🔧 Customization

### Simple Demonstrations
The modular structure makes it easy to customize:

```python
# Customize training parameters
model, optimizer, device = setup_model_and_optimizer(
    temp_dataset, col_names_dict, 
    num_outputs=1, 
    task_type='binary',
    channels=64,  # Increase model capacity
    lr=0.005      # Adjust learning rate
)

# Add custom data preparation
def prepare_custom_data(df):
    """Prepare data for custom task."""
    df['custom_target'] = your_custom_logic(df)
    return df, 'custom_target'

# Use existing training functions
train_loss, train_correct, train_total = train_epoch_binary(
    model, train_loader, optimizer, device
)
```

### Enhanced Examples
```python
# Add to create_enhanced_targets function
df['new_target'] = create_your_target_logic(df)

# Add to task configurations
new_task_config = {
    'new_target': {'type': 'binary', 'column': 'new_target'}
}

# Customize PCA configuration
pca_config = PCAConfig(
    max_components=100,
    variance_thresholds=[0.85, 0.95, 0.99],
    n_clusters=8
)
```

## 📈 Performance

### Simple Demonstrations
- **Quick Training**: 5 epochs per task, ~30 seconds total
- **Educational Focus**: Clear, understandable results
- **Consistent Performance**: Reliable baseline results across runs

### Enhanced Examples
- **Binary Classification**: 80-90% accuracy
- **Regression**: Low MSE with good generalization
- **Multi-class**: 70-85% accuracy depending on task complexity
- **Speed**: CPU ~30-60 seconds, GPU ~15-30 seconds (2-3x speedup)

## 🎓 Learning Path

1. **Start with Simple Demonstrations**: Learn PyTorch Frame basics
   ```bash
   python pytorch_frame_simple_demos.py
   ```

2. **Explore Enhanced Examples**: Advanced features and analysis
   ```bash
   python pytorch_frame_enhanced_examples.py
   ```

3. **Customize and Extend**: Use the modular components for your own projects

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Ensure all tests pass
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch Frame team for the excellent tabular deep learning library
- PyTorch team for the foundational deep learning framework
- scikit-learn team for machine learning utilities 