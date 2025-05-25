"""Enhanced PyTorch Frame examples for binary classification, regression, and multi-class classification task types."""

import os
import time
import torch
import torch.nn.functional as F
from torch_frame import TensorFrame, stype
from torch_frame.data import DataLoader
from torch_frame.datasets import Yandex
from torch_frame.nn import (
    StypeWiseFeatureEncoder,
    TabTransformerConv,
    LinearEncoder,
    EmbeddingEncoder,
)
from torch.nn import Linear, Module, ModuleList, Dropout
import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass, field
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"Running program: {os.path.basename(__file__)}")

@dataclass
class ExperimentConfig:
    sample_size: int = 5000
    epochs: int = 3
    batch_size: int = 256
    learning_rate: float = 0.001
    pca_max_components: int = 50
    anomaly_threshold_percentile: float = 95

@dataclass
class PCAConfig:
    max_components: int = 50
    variance_thresholds: List[float] = field(default_factory=lambda: [0.80, 0.90, 0.95, 0.99])
    n_clusters: int = 5
    anomaly_threshold_percentile: float = 95

def create_enhanced_targets(original_dataset, sample_size=8000):
    """Create diverse targets showcasing different ML problem types."""
    df = original_dataset.df.sample(n=sample_size, random_state=42).copy()
    
    # === BINARY CLASSIFICATION EXAMPLES ===
    # 1. Age-based senior classification (easy)
    df['is_senior'] = (df['N_feature_0'] >= 50).astype(int)
    
    # 2. High earner prediction (medium difficulty)
    df['high_earner'] = (df['target_col'] == 1).astype(int)  # Original income >50K
    
    # 3. Education level threshold (harder)
    df['highly_educated'] = (df['N_feature_2'] >= 13).astype(int)  # Bachelor's+
    
    # === REGRESSION EXAMPLES ===
    # 1. Education years prediction (continuous)
    np.random.seed(42)
    education_noise = np.random.normal(0, 0.8, len(df))
    df['education_years'] = df['N_feature_2'] + education_noise
    
    # 2. Age prediction (different scale)
    age_noise = np.random.normal(0, 3, len(df))
    df['predicted_age'] = df['N_feature_0'] + age_noise
    
    # 3. Normalized capital score (0-1 range)
    capital_total = df['N_feature_3'] + df['N_feature_4']  # capital-gain + capital-loss
    df['capital_score'] = (capital_total - capital_total.min()) / (capital_total.max() - capital_total.min() + 1e-8)
    
    # === CATEGORICAL CLASSIFICATION EXAMPLES ===
    # 1. Work sector (7 classes)
    education_to_sector = {
        'Preschool': 'Service', '1st-4th': 'Manual', '5th-6th': 'Manual', 
        '7th-8th': 'Manual', '9th': 'Service', '10th': 'Service', 
        '11th': 'Service', '12th': 'Service', 'HS-grad': 'Service', 
        'Some-college': 'Administrative', 'Assoc-voc': 'Technical',
        'Assoc-acdm': 'Technical', 'Bachelors': 'Professional', 
        'Masters': 'Management', 'Prof-school': 'Professional', 
        'Doctorate': 'Research'
    }
    df['work_sector'] = df['C_feature_1'].map(education_to_sector).fillna('Other')
    
    # 2. Age group classification (5 classes)
    df['age_group'] = pd.cut(df['N_feature_0'], 
                            bins=[0, 25, 35, 50, 65, 100], 
                            labels=['Young', 'Adult', 'Middle', 'Senior', 'Elderly'])
    
    # 3. Education tier (4 classes)
    education_tier_map = {
        'Preschool': 'Basic', '1st-4th': 'Basic', '5th-6th': 'Basic', '7th-8th': 'Basic',
        '9th': 'Basic', '10th': 'Basic', '11th': 'Basic', '12th': 'Basic', 'HS-grad': 'Basic',
        'Some-college': 'Intermediate', 'Assoc-voc': 'Intermediate', 'Assoc-acdm': 'Intermediate',
        'Bachelors': 'Advanced', 'Masters': 'Expert', 'Prof-school': 'Expert', 'Doctorate': 'Expert'
    }
    df['education_tier'] = df['C_feature_1'].map(education_tier_map).fillna('Basic')
    
    return df


# === MODEL ARCHITECTURES FOR DIFFERENT TASKS ===

class BinaryClassificationModel(Module):
    """Specialized model for binary classification tasks."""
    
    def __init__(self, channels, col_stats, col_names_dict, target_names):
        super().__init__()
        self.target_names = target_names
        
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict={
                stype.numerical: LinearEncoder(),
                stype.categorical: EmbeddingEncoder(),
            },
        )
        
        # Binary-optimized architecture
        self.dropout = Dropout(0.1)
        self.hidden_layer = Linear(channels, channels // 2)
        self.output_heads = ModuleList([Linear(channels // 2, 1) for _ in target_names])

    def forward(self, tf: TensorFrame) -> dict:
        x, _ = self.encoder(tf)
        x_pooled = x.mean(dim=1)
        x_hidden = torch.relu(self.hidden_layer(self.dropout(x_pooled)))
        
        outputs = {}
        for i, target_name in enumerate(self.target_names):
            outputs[target_name] = self.output_heads[i](x_hidden)
        return outputs


class RegressionModel(Module):
    """Specialized model for regression tasks."""
    
    def __init__(self, channels, col_stats, col_names_dict, target_names):
        super().__init__()
        self.target_names = target_names
        
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict={
                stype.numerical: LinearEncoder(),
                stype.categorical: EmbeddingEncoder(),
            },
        )
        
        # Regression-optimized architecture
        self.hidden1 = Linear(channels, channels)
        self.hidden2 = Linear(channels, channels // 2)
        self.dropout = Dropout(0.15)
        self.output_heads = ModuleList([Linear(channels // 2, 1) for _ in target_names])

    def forward(self, tf: TensorFrame) -> dict:
        x, _ = self.encoder(tf)
        x_pooled = x.mean(dim=1)
        x_h1 = torch.relu(self.hidden1(x_pooled))
        x_h2 = torch.relu(self.hidden2(self.dropout(x_h1)))
        
        outputs = {}
        for i, target_name in enumerate(self.target_names):
            outputs[target_name] = self.output_heads[i](x_h2).squeeze(-1)
        return outputs


class MultiClassModel(Module):
    """Specialized model for multi-class classification tasks."""
    
    def __init__(self, channels, col_stats, col_names_dict, target_configs):
        super().__init__()
        self.target_configs = target_configs
        
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict={
                stype.numerical: LinearEncoder(),
                stype.categorical: EmbeddingEncoder(),
            },
        )
        
        # Multi-class optimized with attention-like mechanism
        self.attention = Linear(channels, channels)
        self.dropout = Dropout(0.1)
        self.output_heads = ModuleList()
        
        for target_name, config in target_configs.items():
            self.output_heads.append(Linear(channels, config['num_classes']))

    def forward(self, tf: TensorFrame) -> dict:
        x, _ = self.encoder(tf)
        x_pooled = x.mean(dim=1)
        
        # Apply attention-like weighting
        attention_weights = torch.softmax(self.attention(x_pooled), dim=-1)
        x_attended = x_pooled * attention_weights
        x_attended = self.dropout(x_attended)
        
        outputs = {}
        for i, target_name in enumerate(self.target_configs.keys()):
            outputs[target_name] = self.output_heads[i](x_attended)
        return outputs


def create_tensor_frame(df_split, col_names_dict, target_columns):
    """Create TensorFrame for training."""
    feat_dict = {}
    
    # Numerical features
    if stype.numerical in col_names_dict:
        feat_dict[stype.numerical] = torch.tensor(
            df_split[col_names_dict[stype.numerical]].values, dtype=torch.float32
        )
    
    # Categorical features
    if stype.categorical in col_names_dict:
        categorical_data = []
        for col in col_names_dict[stype.categorical]:
            unique_vals = sorted(df_split[col].unique())
            val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
            encoded = df_split[col].map(val_to_idx).values
            categorical_data.append(encoded)
        feat_dict[stype.categorical] = torch.tensor(
            np.column_stack(categorical_data), dtype=torch.long
        )
    
    # Targets
    y_data = []
    for col in target_columns:
        if col in ['is_senior', 'high_earner', 'highly_educated']:  # Binary
            y_data.append(torch.tensor(df_split[col].values, dtype=torch.long))
        elif col in ['education_years', 'predicted_age', 'capital_score']:  # Regression
            y_data.append(torch.tensor(df_split[col].values, dtype=torch.float32))
        else:  # Categorical
            unique_vals = sorted(df_split[col].unique())
            val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
            encoded = df_split[col].map(val_to_idx).values
            y_data.append(torch.tensor(encoded, dtype=torch.long))
    
    y = torch.stack(y_data, dim=1)
    
    return TensorFrame(feat_dict=feat_dict, y=y, col_names_dict=col_names_dict)


def compute_task_losses(predictions, targets, task_configs):
    """Compute losses for different task types."""
    losses = {}
    for i, (task_name, config) in enumerate(task_configs.items()):
        pred = predictions[task_name]
        target = targets[:, i]
        
        if config['type'] == 'binary':
            losses[task_name] = F.binary_cross_entropy_with_logits(pred.squeeze(), target.float())
        elif config['type'] == 'regression':
            losses[task_name] = F.mse_loss(pred, target.float())
        else:  # categorical
            losses[task_name] = F.cross_entropy(pred, target.long())
    
    return losses


def compute_auc_score(y_true, y_scores):
    """Simple AUC computation."""
    try:
        # Sort by score
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[desc_score_indices]
        
        # Count positive and negative examples
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        # Compute AUC using trapezoidal rule
        auc = 0.0
        tp = 0
        fp = 0
        
        for i in range(len(y_true_sorted)):
            if y_true_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
                auc += tp
        
        return auc / (n_pos * n_neg)
    except:
        return 0.5


def compute_f1_score(y_true, y_pred):
    """Simple F1 score computation."""
    try:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    except:
        return 0.0


def mean_absolute_error(y_true, y_pred):
    """Simple MAE computation."""
    return np.mean(np.abs(y_true - y_pred))


def evaluate_model(model, dataloader, task_configs, device):
    """Comprehensive evaluation with task-specific metrics."""
    model.eval()
    all_predictions = {task: [] for task in task_configs.keys()}
    all_targets = {task: [] for task in task_configs.keys()}
    
    with torch.no_grad():
        for batch_tf in dataloader:
            batch_tf = batch_tf.to(device)
            predictions = model(batch_tf)
            
            for i, (task_name, config) in enumerate(task_configs.items()):
                target = batch_tf.y[:, i]
                pred = predictions[task_name]
                
                if config['type'] == 'binary':
                    prob = torch.sigmoid(pred.squeeze()).cpu()
                    all_predictions[task_name].append(prob)
                    all_targets[task_name].append(target.cpu().float())
                elif config['type'] == 'regression':
                    all_predictions[task_name].append(pred.cpu())
                    all_targets[task_name].append(target.cpu().float())
                else:  # categorical
                    _, predicted = torch.max(pred, 1)
                    all_predictions[task_name].append(predicted.cpu())
                    all_targets[task_name].append(target.cpu())
    
    # Compute metrics
    results = {}
    for task_name, config in task_configs.items():
        preds = torch.cat(all_predictions[task_name])
        targets = torch.cat(all_targets[task_name])
        
        if config['type'] == 'binary':
            binary_preds = (preds > 0.5).float()
            accuracy = (binary_preds == targets).float().mean().item() * 100
            try:
                auc = compute_auc_score(targets.numpy(), preds.numpy())
                f1 = compute_f1_score(targets.numpy(), binary_preds.numpy())
                results[task_name] = {'Accuracy': accuracy, 'AUC': auc, 'F1': f1}
            except:
                results[task_name] = {'Accuracy': accuracy}
        elif config['type'] == 'regression':
            mse = F.mse_loss(preds, targets).item()
            mae = mean_absolute_error(targets.numpy(), preds.numpy())
            rmse = np.sqrt(mse)
            results[task_name] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse}
        else:  # categorical
            accuracy = (preds == targets).float().mean().item() * 100
            try:
                f1 = compute_f1_score(targets.numpy(), preds.numpy())
                results[task_name] = {'Accuracy': accuracy, 'F1': f1}
            except:
                results[task_name] = {'Accuracy': accuracy}
    
    return results


def train_model(model, train_loader, val_loader, task_configs, device, epochs=3, lr=0.001):
    """Train model with comprehensive logging."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    train_history = []
    val_history = []
    
    print(f"Training {model.__class__.__name__} for {epochs} epochs...")
    print("="*80)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = {task: 0.0 for task in task_configs.keys()}
        train_total = 0.0
        num_batches = 0
        
        for batch_tf in train_loader:
            batch_tf = batch_tf.to(device)
            predictions = model(batch_tf)
            
            losses = compute_task_losses(predictions, batch_tf.y, task_configs)
            total_loss = sum(losses.values())
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            for task, loss in losses.items():
                train_losses[task] += loss.item()
            train_total += total_loss.item()
            num_batches += 1
        
        # Validation phase
        model.eval()
        val_losses = {task: 0.0 for task in task_configs.keys()}
        val_total = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_tf in val_loader:
                batch_tf = batch_tf.to(device)
                predictions = model(batch_tf)
                
                losses = compute_task_losses(predictions, batch_tf.y, task_configs)
                total_loss = sum(losses.values())
                
                for task, loss in losses.items():
                    val_losses[task] += loss.item()
                val_total += total_loss.item()
                val_batches += 1
        
        # Calculate averages
        avg_train_total = train_total / num_batches
        avg_val_total = val_total / val_batches
        
        scheduler.step(avg_val_total)
        
        # Store history
        train_epoch = {task: train_losses[task] / num_batches for task in task_configs.keys()}
        val_epoch = {task: val_losses[task] / val_batches for task in task_configs.keys()}
        train_epoch['total'] = avg_train_total
        val_epoch['total'] = avg_val_total
        
        train_history.append(train_epoch)
        val_history.append(val_epoch)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{epochs} | Total - Train: {avg_train_total:.4f}, Val: {avg_val_total:.4f}")
        for task_name, config in task_configs.items():
            train_loss = train_losses[task_name] / num_batches
            val_loss = val_losses[task_name] / val_batches
            print(f"  {task_name:15s} - Train: {train_loss:.4f}, Val: {val_loss:.4f} [{config['type']}]")
    
    return train_history, val_history


def evaluate_binary_classification(enhanced_df, train_df, val_df, col_names_dict, train_dataset, device):
    """Evaluate binary classification models."""
    print(f"\n{'='*80}")
    print("BINARY CLASSIFICATION EXPERIMENTS")
    print("="*80)
    
    binary_tasks = {
        'is_senior': {'type': 'binary', 'description': 'Age >= 50 prediction'},
        'high_earner': {'type': 'binary', 'description': 'Income >50K prediction'},
        'highly_educated': {'type': 'binary', 'description': 'Education >= Bachelor\'s'}
    }
    
    binary_columns = list(binary_tasks.keys())
    train_tf_binary = create_tensor_frame(train_df, col_names_dict, binary_columns)
    val_tf_binary = create_tensor_frame(val_df, col_names_dict, binary_columns)
    
    train_loader_binary = DataLoader(train_tf_binary, batch_size=256, shuffle=True)
    val_loader_binary = DataLoader(val_tf_binary, batch_size=256, shuffle=False)
    
    binary_model = BinaryClassificationModel(
        channels=64, col_stats=train_dataset.col_stats,
        col_names_dict=col_names_dict, target_names=binary_columns
    ).to(device)
    
    print(f"Binary model parameters: {sum(p.numel() for p in binary_model.parameters()):,}")
    
    # Train binary model
    train_hist_binary, val_hist_binary = train_model(
        binary_model, train_loader_binary, val_loader_binary, binary_tasks, device, epochs=3
    )
    
    # Evaluate binary model
    print(f"\nBinary Classification Results:")
    binary_results = evaluate_model(binary_model, val_loader_binary, binary_tasks, device)
    for task_name, metrics in binary_results.items():
        print(f"\n{task_name} ({binary_tasks[task_name]['description']}):")
        for metric, value in metrics.items():
            if metric in ['Accuracy']:
                print(f"  {metric}: {value:.2f}%")
            else:
                print(f"  {metric}: {value:.4f}")
    
    return binary_results, train_hist_binary, val_hist_binary


def evaluate_regression(enhanced_df, train_df, val_df, col_names_dict, train_dataset, device):
    """Evaluate regression models."""
    print(f"\n{'='*80}")
    print("REGRESSION EXPERIMENTS")
    print("="*80)
    
    regression_tasks = {
        'education_years': {'type': 'regression', 'description': 'Years of education'},
        'predicted_age': {'type': 'regression', 'description': 'Age prediction'},
        'capital_score': {'type': 'regression', 'description': 'Capital score (0-1)'}
    }
    
    regression_columns = list(regression_tasks.keys())
    train_tf_reg = create_tensor_frame(train_df, col_names_dict, regression_columns)
    val_tf_reg = create_tensor_frame(val_df, col_names_dict, regression_columns)
    
    train_loader_reg = DataLoader(train_tf_reg, batch_size=256, shuffle=True)
    val_loader_reg = DataLoader(val_tf_reg, batch_size=256, shuffle=False)
    
    regression_model = RegressionModel(
        channels=64, col_stats=train_dataset.col_stats,
        col_names_dict=col_names_dict, target_names=regression_columns
    ).to(device)
    
    print(f"Regression model parameters: {sum(p.numel() for p in regression_model.parameters()):,}")
    
    # Train regression model
    train_hist_reg, val_hist_reg = train_model(
        regression_model, train_loader_reg, val_loader_reg, regression_tasks, device, epochs=3
    )
    
    # Evaluate regression model
    print(f"\nRegression Results:")
    regression_results = evaluate_model(regression_model, val_loader_reg, regression_tasks, device)
    for task_name, metrics in regression_results.items():
        print(f"\n{task_name} ({regression_tasks[task_name]['description']}):")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return regression_results, train_hist_reg, val_hist_reg


def evaluate_multiclass_classification(enhanced_df, train_df, val_df, col_names_dict, train_dataset, device):
    """Evaluate multi-class classification models."""
    print(f"\n{'='*80}")
    print("MULTI-CLASS CLASSIFICATION EXPERIMENTS")
    print("="*80)
    
    multiclass_tasks = {
        'work_sector': {
            'type': 'categorical', 
            'num_classes': len(enhanced_df['work_sector'].unique()),
            'description': 'Work sector classification (7 classes)'
        },
        'age_group': {
            'type': 'categorical', 
            'num_classes': len(enhanced_df['age_group'].unique()),
            'description': 'Age group classification (5 classes)'
        },
        'education_tier': {
            'type': 'categorical', 
            'num_classes': len(enhanced_df['education_tier'].unique()),
            'description': 'Education tier classification (4 classes)'
        }
    }
    
    multiclass_columns = list(multiclass_tasks.keys())
    train_tf_multi = create_tensor_frame(train_df, col_names_dict, multiclass_columns)
    val_tf_multi = create_tensor_frame(val_df, col_names_dict, multiclass_columns)
    
    train_loader_multi = DataLoader(train_tf_multi, batch_size=256, shuffle=True)
    val_loader_multi = DataLoader(val_tf_multi, batch_size=256, shuffle=False)
    
    multiclass_model = MultiClassModel(
        channels=64, col_stats=train_dataset.col_stats,
        col_names_dict=col_names_dict, target_configs=multiclass_tasks
    ).to(device)
    
    print(f"Multi-class model parameters: {sum(p.numel() for p in multiclass_model.parameters()):,}")
    
    # Train multi-class model
    train_hist_multi, val_hist_multi = train_model(
        multiclass_model, train_loader_multi, val_loader_multi, multiclass_tasks, device, epochs=3
    )
    
    # Evaluate multi-class model
    print(f"\nMulti-Class Classification Results:")
    multiclass_results = evaluate_model(multiclass_model, val_loader_multi, multiclass_tasks, device)
    for task_name, metrics in multiclass_results.items():
        print(f"\n{task_name} ({multiclass_tasks[task_name]['description']}):")
        for metric, value in metrics.items():
            if metric in ['Accuracy']:
                print(f"  {metric}: {value:.2f}%")
            else:
                print(f"  {metric}: {value:.4f}")
    
    return multiclass_results, train_hist_multi, val_hist_multi


def run_demo_with_device(device_name):
    """Run the enhanced examples on a specific device."""
    start_time = time.time()
    
    print("="*80)
    print(f"PYTORCH FRAME ENHANCED TRAINING EXAMPLES (Device: {device_name})")
    print("="*80)
    
    # Set device
    device = torch.device(device_name)
    
    # Load and prepare data FIRST
    print("\n1. Loading and preparing enhanced dataset...")
    dataset = Yandex(root="./data/adult", name="adult")
    dataset.materialize()
    
    # Use a smaller sample size for faster testing
    sample_size = 5000
    enhanced_df = create_enhanced_targets(dataset, sample_size=sample_size)
    print(f"   Dataset size: {len(enhanced_df)} samples")
    
    # Get feature structure
    train_dataset = dataset[:0.8]
    col_names_dict = train_dataset.tensor_frame.col_names_dict
    
    # === PCA ANALYSIS (now that variables are defined) ===
    pca_target_configs = {
        'is_senior': {'type': 'binary', 'column': 'is_senior'},
        'high_earner': {'type': 'binary', 'column': 'high_earner'},
        'highly_educated': {'type': 'binary', 'column': 'highly_educated'},
        'education_years': {'type': 'regression', 'column': 'education_years'},
        'predicted_age': {'type': 'regression', 'column': 'predicted_age'},
        'capital_score': {'type': 'regression', 'column': 'capital_score'},
        'work_sector': {'type': 'categorical', 'column': 'work_sector'},
        'age_group': {'type': 'categorical', 'column': 'age_group'},
        'education_tier': {'type': 'categorical', 'column': 'education_tier'}
    }
    if device_name == 'cpu':
        print("\n=== PERFORMING PCA ANALYSIS ===")
        pca_results = perform_comprehensive_pca_analysis(enhanced_df, col_names_dict, pca_target_configs)
    
    # Prepare data splits
    feature_columns = col_names_dict[stype.numerical] + col_names_dict[stype.categorical]
    
    train_size = int(0.7 * len(enhanced_df))
    val_size = int(0.15 * len(enhanced_df))
    
    train_df = enhanced_df[:train_size].copy()
    val_df = enhanced_df[train_size:train_size + val_size].copy()
    test_df = enhanced_df[train_size + val_size:].copy()
    
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)} samples")
    print(f"   Device: {device}")
    
    # === BINARY CLASSIFICATION EXPERIMENTS ===
    binary_results, train_hist_binary, val_hist_binary = evaluate_binary_classification(enhanced_df, train_df, val_df, col_names_dict, train_dataset, device)
    
    # === REGRESSION EXPERIMENTS ===
    regression_results, train_hist_reg, val_hist_reg = evaluate_regression(enhanced_df, train_df, val_df, col_names_dict, train_dataset, device)
    
    # === MULTI-CLASS CLASSIFICATION EXPERIMENTS ===
    multiclass_results, train_hist_multi, val_hist_multi = evaluate_multiclass_classification(enhanced_df, train_df, val_df, col_names_dict, train_dataset, device)
    
    # Calculate and return execution time
    execution_time = time.time() - start_time
    return execution_time


def demonstrate_string_to_ohe_conversion():
    """Demonstrate converting string columns to One-Hot Encoded format using Polars and PyTorch."""
    print("\n" + "="*80)
    print("STRING TO ONE-HOT ENCODING (OHE) CONVERSION DEMONSTRATION")
    print("="*80)
    
    # Create sample DataFrame with string columns
    sample_data = {
        "category": ["apple", "banana", "apple", "orange", "banana", "cherry", "apple", "orange"],
        "color": ["red", "yellow", "green", "orange", "yellow", "red", "red", "orange"],
        "size": ["small", "medium", "small", "large", "medium", "small", "medium", "large"],
        "numeric_feature": [1.2, 2.3, 1.1, 3.4, 2.1, 1.5, 1.8, 3.2]
    }
    
    # Create Polars DataFrame
    df = pl.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    # Define string columns to convert
    string_columns = ["category", "color", "size"]
    
    print(f"\nConverting string columns to OHE: {string_columns}")
    
    # Convert each string column to OHE
    ohe_results = {}
    category_mappings = {}
    
    for col in string_columns:
        print(f"\n--- Processing column: '{col}' ---")
        
        # Step 1: Convert to categorical and get indices
        df_with_indices = df.with_columns(
            pl.col(col).cast(pl.Categorical).to_physical().alias(f"{col}_idx")
        )
        
        # Step 2: Get unique categories and create mapping
        categories = df[col].unique().to_list()
        num_categories = len(categories)
        category_mappings[col] = {i: cat for i, cat in enumerate(categories)}
        
        print(f"Categories found: {categories}")
        print(f"Number of categories: {num_categories}")
        print(f"Category mapping: {category_mappings[col]}")
        
        # Step 3: Convert indices to PyTorch tensor
        indices_tensor = torch.tensor(df_with_indices[f"{col}_idx"].to_list(), dtype=torch.long)
        
        # Step 4: Create one-hot encoding
        one_hot = torch.zeros(indices_tensor.size(0), num_categories)
        one_hot.scatter_(1, indices_tensor.unsqueeze(1), 1)
        
        ohe_results[col] = {
            'tensor': one_hot,
            'categories': categories,
            'mapping': category_mappings[col],
            'indices': indices_tensor
        }
        
        print(f"Original values: {df[col].to_list()}")
        print(f"Categorical indices: {indices_tensor.tolist()}")
        print(f"One-Hot Encoded tensor shape: {one_hot.shape}")
        print(f"One-Hot Encoded tensor:")
        print(one_hot)
    
    # Demonstrate combining OHE features with numeric features
    print(f"\n--- Combining OHE features with numeric features ---")
    
    # Get numeric feature
    numeric_tensor = torch.tensor(df["numeric_feature"].to_list(), dtype=torch.float32).unsqueeze(1)
    print(f"Numeric feature tensor shape: {numeric_tensor.shape}")
    
    # Concatenate all OHE features
    all_ohe_features = torch.cat([ohe_results[col]['tensor'] for col in string_columns], dim=1)
    print(f"Combined OHE features shape: {all_ohe_features.shape}")
    
    # Combine OHE with numeric features
    combined_features = torch.cat([numeric_tensor, all_ohe_features], dim=1)
    print(f"Final combined feature tensor shape: {combined_features.shape}")
    print(f"Combined features tensor:")
    print(combined_features)
    
    # Demonstrate using the combined features in a simple neural network
    print(f"\n--- Using OHE features in a neural network ---")
    
    class SimpleOHEModel(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim=16, output_dim=1):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
            self.dropout = torch.nn.Dropout(0.1)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # Create model
    input_dim = combined_features.shape[1]
    model = SimpleOHEModel(input_dim)
    
    print(f"Model input dimension: {input_dim}")
    print(f"Model architecture:")
    print(model)
    
    # Forward pass demonstration
    with torch.no_grad():
        output = model(combined_features)
        print(f"Model output shape: {output.shape}")
        print(f"Sample predictions: {output.squeeze().tolist()}")
    
    # Summary
    print(f"\n--- OHE Conversion Summary ---")
    total_ohe_features = sum(len(ohe_results[col]['categories']) for col in string_columns)
    print(f"Original string columns: {len(string_columns)}")
    print(f"Total OHE features created: {total_ohe_features}")
    print(f"Feature breakdown:")
    for col in string_columns:
        print(f"  {col}: {len(ohe_results[col]['categories'])} features")
    print(f"Numeric features: 1")
    print(f"Total combined features: {input_dim}")
    
    return ohe_results, combined_features, model


def standardize_features(features):
    """Standardize features for PCA (mean=0, std=1)."""
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)
    return standardized_features, scaler


def analyze_pca_variance(features, max_components=50):
    """Analyze how many components explain variance."""
    
    pca = PCA(n_components=min(max_components, features.shape[1]))
    pca.fit(features)
    
    # Calculate cumulative variance explained
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find components for different variance thresholds
    thresholds = [0.80, 0.90, 0.95, 0.99]
    components_needed = {}
    for threshold in thresholds:
        idx = np.argmax(cumulative_variance >= threshold)
        components_needed[threshold] = idx + 1
    
    return pca, cumulative_variance, components_needed


def analyze_feature_importance_in_pcs(pca, feature_names, top_n_components=10):
    """Analyze which original features contribute most to each PC."""
    
    feature_importance = {}
    for i in range(min(top_n_components, pca.n_components_)):
        # Get absolute loadings for this component
        loadings = np.abs(pca.components_[i])
        
        # Get top contributing features
        top_indices = np.argsort(loadings)[-10:][::-1]
        top_features = [(feature_names[idx], loadings[idx]) for idx in top_indices]
        
        feature_importance[f'PC{i+1}'] = {
            'variance_explained': pca.explained_variance_ratio_[i],
            'top_features': top_features
        }
    
    return feature_importance


def analyze_target_separability(pca_features, enhanced_df, target_configs, n_components=10):
    """Analyze how well targets can be separated in PCA space."""
    
    separability_results = {}
    
    for target_name, config in target_configs.items():
        try:
            target_values = enhanced_df[config['column']]
            
            if config['type'] in ['binary', 'categorical']:
                # Calculate silhouette score for different numbers of components
                silhouette_scores = []
                for n_comp in range(2, min(n_components + 1, pca_features.shape[1])):
                    features_subset = pca_features[:, :n_comp]
                    try:
                        score = silhouette_score(features_subset, target_values)
                        silhouette_scores.append((n_comp, score))
                    except:
                        silhouette_scores.append((n_comp, 0))
                
                separability_results[target_name] = {
                    'type': config['type'],
                    'silhouette_scores': silhouette_scores,
                    'best_components': max(silhouette_scores, key=lambda x: x[1]) if silhouette_scores else (2, 0)
                }
        except KeyError as e:
            logger.warning(f"Target column '{config['column']}' not found for {target_name}: {e}")
            continue
        except ValueError as e:
            logger.warning(f"Invalid data for target {target_name}: {e}")
            continue
    
    return separability_results


def create_pca_enhanced_features(features, n_components_dict):
    """Create different PCA feature sets for different targets."""
    
    pca_features = {}
    pca_models = {}
    
    for target_name, n_components in n_components_dict.items():
        pca = PCA(n_components=n_components)
        transformed_features = pca.fit_transform(features)
        
        pca_features[target_name] = transformed_features
        pca_models[target_name] = pca
    
    return pca_features, pca_models


def analyze_feature_clusters(pca, feature_names, n_clusters=5):
    """Cluster original features based on PCA loadings."""
    
    # Use loadings from first few components
    loadings_matrix = pca.components_[:10].T  # Features x Components
    
    # Cluster features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    feature_clusters = kmeans.fit_predict(loadings_matrix)
    
    # Group features by cluster
    clustered_features = {}
    for i in range(n_clusters):
        cluster_features = [feature_names[j] for j, cluster in enumerate(feature_clusters) if cluster == i]
        clustered_features[f'Cluster_{i+1}'] = cluster_features
    
    return clustered_features, feature_clusters


def detect_anomalies_with_pca(pca_features, threshold_percentile=95):
    """Detect potential anomalies using PCA reconstruction error."""
    
    # Calculate reconstruction error
    reconstruction_errors = np.sum((pca_features - np.mean(pca_features, axis=0))**2, axis=1)
    
    # Define threshold
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    
    # Identify anomalies
    anomaly_indices = np.where(reconstruction_errors > threshold)[0]
    
    return anomaly_indices, reconstruction_errors, threshold


def create_pca_tensor_frame(df_split, col_names_dict, config):
    """Create TensorFrame for PCA features (only numerical)."""
    feat_dict = {}
    
    # PCA features are all numerical
    feat_dict[stype.numerical] = torch.tensor(
        df_split[col_names_dict[stype.numerical]].values, dtype=torch.float32
    )
    
    # Target
    target_col = config['column']
    if config['type'] == 'regression':
        y = torch.tensor(df_split[target_col].values, dtype=torch.float32)
    else:  # binary or categorical
        if target_col in df_split.columns:
            if config['type'] == 'binary':
                y = torch.tensor(df_split[target_col].values, dtype=torch.long)
            else:  # categorical
                unique_vals = sorted(df_split[target_col].unique())
                val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
                encoded = df_split[target_col].map(val_to_idx).values
                y = torch.tensor(encoded, dtype=torch.long)
        else:
            y = torch.zeros(len(df_split), dtype=torch.long)
    
    return TensorFrame(feat_dict=feat_dict, y=y, col_names_dict=col_names_dict)


def compute_loss(predictions, targets, task_type):
    """Compute loss for different task types."""
    if task_type == 'binary':
        return F.binary_cross_entropy_with_logits(predictions, targets.float())
    elif task_type == 'regression':
        return F.mse_loss(predictions, targets.float())
    else:  # categorical
        return F.cross_entropy(predictions, targets.long())


def evaluate_pca_model(model, dataloader, config, device):
    """Evaluate PCA model performance."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            predictions = model(batch)
            
            if config['type'] == 'binary':
                prob = torch.sigmoid(predictions).cpu()
                all_predictions.append(prob)
                all_targets.append(batch.y.cpu().float())
            elif config['type'] == 'regression':
                all_predictions.append(predictions.cpu())
                all_targets.append(batch.y.cpu().float())
            else:  # categorical
                _, predicted = torch.max(predictions, 1)
                all_predictions.append(predicted.cpu())
                all_targets.append(batch.y.cpu())
    
    # Compute metrics
    preds = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    
    results = {}
    if config['type'] == 'binary':
        binary_preds = (preds > 0.5).float()
        accuracy = (binary_preds == targets).float().mean().item() * 100
        results['accuracy'] = accuracy
    elif config['type'] == 'regression':
        mse = F.mse_loss(preds, targets).item()
        mae = torch.mean(torch.abs(preds - targets)).item()
        results['mse'] = mse
        results['mae'] = mae
    else:  # categorical
        accuracy = (preds == targets).float().mean().item() * 100
        results['accuracy'] = accuracy
    
    return results


def train_with_original_features(enhanced_df, col_names_dict, target_name, config, col_stats, device):
    """Train model with original features (placeholder implementation)."""
    print(f"Training model for {target_name} with original features...")
    
    try:
        # Create model with proper parameters
        model = create_model_for_task(config, col_stats, col_names_dict)
        model = model.to(device)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(enhanced_df, col_names_dict, config)
        
        if train_loader is None or val_loader is None:
            print(f"Warning: Could not create data loaders for {target_name}")
            return {'error': 'Failed to create data loaders'}
        
        # Create task configs for training
        task_configs = {target_name: config}
        
        # Train model
        start_time = time.time()
        train_history, val_history = train_model(model, train_loader, val_loader, task_configs, device, epochs=2)
        training_time = time.time() - start_time
        
        # Evaluate model
        results = evaluate_model(model, val_loader, task_configs, device)
        
        # Add metadata
        if target_name in results:
            results[target_name]['training_time'] = training_time
            results[target_name]['features'] = len(col_names_dict[stype.numerical]) + len(col_names_dict[stype.categorical])
        
        return results
        
    except Exception as e:
        print(f"Error training model for {target_name}: {e}")
        return {'error': str(e)}


def perform_comprehensive_pca_analysis(enhanced_df, col_names_dict, target_configs):
    """Main function to perform comprehensive PCA analysis."""
    
    print("="*80)
    print("COMPREHENSIVE PCA ANALYSIS FOR DATASET UNDERSTANDING")
    print("="*80)
    
    # Phase 1: Prepare features
    print("\n1. Preparing features for PCA analysis...")
    features, feature_names = prepare_features_for_pca(enhanced_df, col_names_dict)
    standardized_features, scaler = standardize_features(features)
    print(f"   Total features: {len(feature_names)}")
    print(f"   Numerical features: {len(col_names_dict[stype.numerical])}")
    print(f"   Categorical features (OHE): {len(feature_names) - len(col_names_dict[stype.numerical])}")
    
    # Phase 2: PCA analysis
    print("\n2. Performing PCA variance analysis...")
    pca, cumulative_variance, components_needed = analyze_pca_variance(standardized_features)
    
    print("   Variance explained by component count:")
    for threshold, n_comp in components_needed.items():
        print(f"     {threshold*100:.0f}% variance: {n_comp} components")
    
    feature_importance = analyze_feature_importance_in_pcs(pca, feature_names)
    
    print("\n   Top contributing features to first 3 PCs:")
    for i in range(min(3, len(feature_importance))):
        pc_name = f'PC{i+1}'
        if pc_name in feature_importance:
            print(f"     {pc_name} ({feature_importance[pc_name]['variance_explained']:.3f} variance):")
            for feat_name, loading in feature_importance[pc_name]['top_features'][:3]:
                print(f"       {feat_name}: {loading:.3f}")
    
    # Phase 3: Target-aware analysis
    print("\n3. Analyzing target separability in PCA space...")
    pca_features = pca.transform(standardized_features)
    separability_results = analyze_target_separability(pca_features, enhanced_df, target_configs)
    
    for target_name, result in separability_results.items():
        if 'best_components' in result:
            best_n, best_score = result['best_components']
            print(f"   {target_name}: Best separability with {best_n} components (score: {best_score:.3f})")
    
    # Phase 4: Visualization
    print("\n4. Creating PCA visualizations...")
    try:
        fig = visualize_pca_by_targets(pca_features, enhanced_df, target_configs)
        print("   PCA scatter plots created successfully")
    except Exception as e:
        print(f"   Visualization failed: {e}")
        fig = None
    
    # Phase 5: Feature clustering
    print("\n5. Analyzing feature clusters...")
    clustered_features, feature_clusters = analyze_feature_clusters(pca, feature_names)
    
    print("   Feature clusters:")
    for cluster_name, features in clustered_features.items():
        print(f"     {cluster_name}: {len(features)} features")
        if features:
            print(f"       Examples: {', '.join(features[:3])}")
    
    # Phase 6: Anomaly detection
    print("\n6. Detecting anomalies using PCA...")
    anomaly_indices, reconstruction_errors, threshold = detect_anomalies_with_pca(pca_features)
    print(f"   Detected {len(anomaly_indices)} potential anomalies ({len(anomaly_indices)/len(enhanced_df)*100:.1f}%)")
    print(f"   Reconstruction error threshold: {threshold:.3f}")
    
    return {
        'pca_model': pca,
        'pca_features': pca_features,
        'feature_importance': feature_importance,
        'separability_results': separability_results,
        'clustered_features': clustered_features,
        'anomaly_info': (anomaly_indices, reconstruction_errors, threshold),
        'variance_info': (cumulative_variance, components_needed),
        'visualization': fig,
        'feature_names': feature_names,
        'scaler': scaler
    }


def visualize_pca_by_targets(pca_features, enhanced_df, target_configs):
    """Create PCA scatter plots colored by different targets."""
    
    # Use first 2 principal components for visualization
    pc1, pc2 = pca_features[:, 0], pca_features[:, 1]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    for target_name, config in target_configs.items():
        if plot_idx >= 6:  # Limit to 6 plots
            break
            
        ax = axes[plot_idx]
        target_values = enhanced_df[config['column']]
        
        if config['type'] == 'binary':
            colors = ['red' if x == 1 else 'blue' for x in target_values]
            ax.scatter(pc1, pc2, c=colors, alpha=0.6, s=20)
            ax.set_title(f'{target_name} (Binary)\nRed=1, Blue=0')
        elif config['type'] == 'regression':
            scatter = ax.scatter(pc1, pc2, c=target_values, alpha=0.6, s=20, cmap='viridis')
            plt.colorbar(scatter, ax=ax)
            ax.set_title(f'{target_name} (Regression)')
        else:  # categorical
            unique_vals = target_values.unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_vals)))
            for i, val in enumerate(unique_vals):
                mask = target_values == val
                ax.scatter(pc1[mask], pc2[mask], c=[colors[i]], 
                          label=str(val), alpha=0.6, s=20)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_title(f'{target_name} (Categorical)')
        
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def prepare_features_for_pca(enhanced_df, col_names_dict):
    """Prepare all features for PCA analysis."""
    
    # Extract numerical features directly
    numerical_features = enhanced_df[col_names_dict[stype.numerical]].values
    
    # One-hot encode categorical features
    categorical_ohe = []
    for col in col_names_dict[stype.categorical]:
        ohe = pd.get_dummies(enhanced_df[col], prefix=col)
        categorical_ohe.append(ohe)
    
    # Combine all features
    if categorical_ohe:
        categorical_combined = pd.concat(categorical_ohe, axis=1)
        all_features = np.hstack([numerical_features, categorical_combined.values])
        feature_names = (col_names_dict[stype.numerical] + 
                        list(categorical_combined.columns))
    else:
        all_features = numerical_features
        feature_names = col_names_dict[stype.numerical]
    
    return all_features, feature_names


def create_model_for_task(config, col_stats, col_names_dict):
    """Create appropriate model based on task configuration with proper parameters."""
    if config['type'] == 'binary':
        return BinaryClassificationModel(64, col_stats, col_names_dict, [config['column']])
    elif config['type'] == 'regression':
        return RegressionModel(64, col_stats, col_names_dict, [config['column']])
    else:
        return MultiClassModel(64, col_stats, col_names_dict, {config['column']: config})


def create_data_loaders(enhanced_df, col_names_dict, config):
    """Create train/val data loaders for a specific task."""
    try:
        # Create basic data splits
        train_size = int(0.7 * len(enhanced_df))
        val_size = int(0.15 * len(enhanced_df))
        
        train_df = enhanced_df[:train_size].copy()
        val_df = enhanced_df[train_size:train_size + val_size].copy()
        
        # Create tensor frames
        train_tf = create_tensor_frame(train_df, col_names_dict, [config['column']])
        val_tf = create_tensor_frame(val_df, col_names_dict, [config['column']])
        
        # Create data loaders
        train_loader = DataLoader(train_tf, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_tf, batch_size=256, shuffle=False)
        
        return train_loader, val_loader
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        return None, None


# Run tests on both CPU and GPU if available
if __name__ == "__main__":
    # Run on CPU
    cpu_time = run_demo_with_device('cpu')
    
    # Run on GPU if available
    gpu_time = None
    if torch.cuda.is_available():
        # Clear CUDA cache to ensure fair comparison
        torch.cuda.empty_cache()
        gpu_time = run_demo_with_device('cuda')
    
    # Print summary
    print("\nPerformance Summary for {}:".format(os.path.basename(__file__)))
    print(f"CPU execution time: {cpu_time:.2f} seconds")
    if gpu_time:
        print(f"GPU execution time: {gpu_time:.2f} seconds")
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"GPU speedup: {speedup:.2f}x faster than CPU")
    else:
        print("GPU not available for testing")

def safe_pca_analysis(features, config):
    """Safely perform PCA analysis with error handling."""
    try:
        return analyze_pca_variance(features, config.max_components)
    except ValueError as e:
        logger.error(f"PCA analysis failed: {e}")
        return None, None, {} 