"""Enhanced PyTorch Frame examples with comprehensive training scenarios for different task types."""

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

print(f"Running program: {os.path.basename(__file__)}")

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


def run_demo_with_device(device_name):
    """Run the enhanced examples on a specific device."""
    start_time = time.time()
    
    print("="*80)
    print(f"PYTORCH FRAME ENHANCED TRAINING EXAMPLES (Device: {device_name})")
    print("="*80)
    
    # Set device
    device = torch.device(device_name)
    
    # Load and prepare data
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
    
    # === REGRESSION EXPERIMENTS ===
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
    
    # === MULTI-CLASS CLASSIFICATION EXPERIMENTS ===
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
    
    # Calculate and return execution time
    execution_time = time.time() - start_time
    return execution_time


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