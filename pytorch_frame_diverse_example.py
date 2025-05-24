"""PyTorch Frame multi-task example with diverse target types."""

import os
import time
import polars as pl
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
from torch.nn import Linear, Module, ModuleList, Embedding, MSELoss, BCEWithLogitsLoss
import numpy as np
import pandas as pd

print(f"Running program: {os.path.basename(__file__)}")

def create_multi_target_dataset(original_dataset):
    """Create additional targets from the Adult dataset for multi-task learning."""
    # Get the original dataframe
    df = original_dataset.df.copy()
    
    # Target 1: Original categorical target (income >50K) - kept as is
    # This will be our multi-class categorical target
    
    # Target 2: Boolean target - whether person is senior (age >= 50)
    # N_feature_0 appears to be age based on the statistics (mean ~38.6)
    df['is_senior'] = (df['N_feature_0'] >= 50).astype(int)
    
    # Target 3: Regression target - predict approximate years of education
    # N_feature_2 appears to be education level (mean ~10.0), let's use it directly
    # and add some noise to make it more realistic for regression
    np.random.seed(42)
    noise = np.random.normal(0, 1, len(df))
    df['education_years'] = df['N_feature_2'] + noise
    
    # Target 4: Multi-class categorical - work sector classification
    # Based on C_feature_1 (education level) we'll create work sectors
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
    
    return df


def compute_multi_task_loss(predictions: dict, targets: dict, target_config: dict) -> dict:
    """Compute losses for all tasks."""
    losses = {}
    
    for task_name, pred in predictions.items():
        target = targets[task_name]
        config = target_config[task_name]
        
        if config['type'] == 'regression':
            loss_fn = F.mse_loss
            losses[task_name] = loss_fn(pred, target.float())
        elif config['type'] == 'binary':
            loss_fn = F.binary_cross_entropy_with_logits
            losses[task_name] = loss_fn(pred.squeeze(), target.float())
        else:  # categorical
            loss_fn = F.cross_entropy
            losses[task_name] = loss_fn(pred, target.long())
    
    return losses


def evaluate_multi_task(model: Module, dataloader: DataLoader, target_config: dict, device: torch.device) -> dict:
    """Evaluate model on all tasks."""
    model.eval()
    metrics = {task: {'correct': 0, 'total': 0, 'loss': 0.0} for task in target_config.keys()}
    
    with torch.no_grad():
        for batch_tf in dataloader:
            batch_tf = batch_tf.to(device)
            predictions = model(batch_tf)
            
            # Extract targets
            targets = {}
            for i, (task_name, config) in enumerate(target_config.items()):
                targets[task_name] = batch_tf.y[:, i]
            
            # Compute losses
            losses = compute_multi_task_loss(predictions, targets, target_config)
            
            # Update metrics
            for task_name, pred in predictions.items():
                config = target_config[task_name]
                target = targets[task_name]
                
                metrics[task_name]['loss'] += losses[task_name].item()
                metrics[task_name]['total'] += target.size(0)
                
                if config['type'] == 'regression':
                    # For regression, we can compute MAE or RMSE
                    continue
                elif config['type'] == 'binary':
                    predicted = (torch.sigmoid(pred.squeeze()) > 0.5).float()
                    metrics[task_name]['correct'] += (predicted == target.float()).sum().item()
                else:  # categorical
                    _, predicted = torch.max(pred, 1)
                    metrics[task_name]['correct'] += (predicted == target.long()).sum().item()
    
    # Compute final metrics
    results = {}
    for task_name, metric in metrics.items():
        config = target_config[task_name]
        results[task_name] = {
            'loss': metric['loss'] / len(dataloader),
        }
        if config['type'] != 'regression':
            results[task_name]['accuracy'] = 100.0 * metric['correct'] / metric['total']
    
    return results


# Define the 3. Multi-Task Tabular Model class
class MultiTaskTabularModel(Module):
    """Tabular model with multiple output heads for different task types."""

    def __init__(
        self,
        channels: int,
        num_transformer_layers: int,
        num_heads: int,
        col_stats: dict,
        col_names_dict: dict,
        target_config: dict,
    ):
        super().__init__()
        self.target_config = target_config
        
        # Shared encoder
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict={
                stype.numerical: LinearEncoder(),
                stype.categorical: EmbeddingEncoder(),
            },
        )

        # Shared transformer layers
        self.convs = ModuleList()
        for _ in range(num_transformer_layers):
            self.convs.append(
                TabTransformerConv(
                    channels=channels,
                    num_heads=num_heads,
                    attn_dropout=0.0,
                    ffn_dropout=0.0,
                )
            )

        # Task-specific output heads
        self.output_heads = ModuleList()
        for task_name, config in target_config.items():
            if config['type'] == 'regression':
                head = Linear(channels, 1)
            elif config['type'] == 'binary':
                head = Linear(channels, 1)  # Binary classification needs 1 output (logit)
            else:  # categorical or binary
                head = Linear(channels, config['num_classes'])
            self.output_heads.append(head)

    def forward(self, tf: TensorFrame) -> dict:
        """Forward pass returning predictions for all tasks."""
        # Shared representation
        x, _ = self.encoder(tf)
        for conv in self.convs:
            x = conv(x)
        x_pooled = x.mean(dim=1)

        # Task-specific predictions
        outputs = {}
        for i, (task_name, config) in enumerate(self.target_config.items()):
            pred = self.output_heads[i](x_pooled)
            if config['type'] == 'regression':
                pred = pred.squeeze(-1)  # Remove last dimension for regression
            outputs[task_name] = pred
            
        return outputs


def create_multi_task_tensor_frame(df_split, col_names_dict, target_config):
    """Create a TensorFrame with multiple targets."""
    from torch_frame import TensorFrame
    
    # Prepare the data dictionary
    feat_dict = {}
    
    # Add numerical features
    if stype.numerical in col_names_dict:
        numerical_data = torch.tensor(
            df_split[col_names_dict[stype.numerical]].values, 
            dtype=torch.float32
        )
        feat_dict[stype.numerical] = numerical_data
    
    # Add categorical features  
    if stype.categorical in col_names_dict:
        categorical_data = []
        for col in col_names_dict[stype.categorical]:
            # Convert categorical values to indices
            unique_vals = sorted(df_split[col].unique())
            val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
            encoded = df_split[col].map(val_to_idx).values
            categorical_data.append(encoded)
        
        categorical_tensor = torch.tensor(
            np.column_stack(categorical_data), 
            dtype=torch.long
        )
        feat_dict[stype.categorical] = categorical_tensor
    
    # Prepare targets (stack all target columns)
    target_tensors = []
    for task_name, config in target_config.items():
        col = config['column']
        if config['type'] == 'regression':
            target_tensor = torch.tensor(df_split[col].values, dtype=torch.float32)
        elif config['type'] == 'binary':
            target_tensor = torch.tensor(df_split[col].values, dtype=torch.long)
        else:  # categorical
            if col == 'target_col':
                # Use the original target values directly
                target_tensor = torch.tensor(df_split[col].values, dtype=torch.long)
            else:
                # For other categorical targets, map to indices
                unique_vals = sorted(df_split[col].unique())
                val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
                encoded = df_split[col].map(val_to_idx).values
                target_tensor = torch.tensor(encoded, dtype=torch.long)
        target_tensors.append(target_tensor)
    
    y = torch.stack(target_tensors, dim=1)
    
    return TensorFrame(
        feat_dict=feat_dict,
        y=y,
        col_names_dict=col_names_dict,
    )


def run_demo(device_name='cpu'):
    """Run the demo on a specific device."""
    start_time = time.time()
    
    print("="*60)
    print(f"PyTorch Frame Multi-Task Example (Device: {device_name})")
    print("="*60)
    
    # Ensure device is valid
    device = torch.device(device_name)
    
    # 1. Load dataset and create multiple targets
    print("Loading dataset and creating multi-task targets...")
    dataset = Yandex(root="./data/adult", name="adult")
    dataset.materialize()
    
    # Create enhanced dataset with multiple targets
    enhanced_df = create_multi_target_dataset(dataset)
    
    # Define our multiple targets with different types
    target_config = {
        'income_category': {  # Original categorical target (>50K income)
            'column': 'target_col',  # Use the actual column name from the dataframe
            'type': 'categorical',
            'num_classes': dataset.num_classes
        },
        'is_senior': {  # Boolean target (age >= 50)
            'column': 'is_senior', 
            'type': 'binary',
            'num_classes': 2
        },
        'education_years': {  # Regression target (years of education)
            'column': 'education_years',
            'type': 'regression',
            'num_classes': 1
        },
        'work_sector': {  # Categorical target (work sector)
            'column': 'work_sector',
            'type': 'categorical', 
            'num_classes': len(enhanced_df['work_sector'].unique())
        }
    }
    
    print(f"Multi-task targets created:")
    for name, config in target_config.items():
        print(f"  {name}: {config['type']} ({config['num_classes']} classes/dims)")
    
    # Get the actual column structure from the dataset
    train_dataset = dataset[:0.8]
    actual_col_names_dict = train_dataset.tensor_frame.col_names_dict
    
    print(f"Available features:")
    print(f"  Numerical: {actual_col_names_dict[stype.numerical]}")
    print(f"  Categorical: {actual_col_names_dict[stype.categorical]}")
    
    # Use all available features from the dataset
    feature_columns = (actual_col_names_dict[stype.numerical] + 
                      actual_col_names_dict[stype.categorical])
    
    # Filter dataframe to selected features + targets
    target_columns = [cfg['column'] for cfg in target_config.values()]
    df_filtered = enhanced_df[feature_columns + target_columns].copy()
    
    # Convert to pandas and handle missing values
    df_pandas = df_filtered.to_pandas() if hasattr(df_filtered, 'to_pandas') else df_filtered
    df_pandas = df_pandas.fillna('Unknown')
    
    print(f"Selected features: {len(feature_columns)} features")
    print(f"  Numerical: {len(actual_col_names_dict[stype.numerical])}")
    print(f"  Categorical: {len(actual_col_names_dict[stype.categorical])}")
    
    # Split data
    train_size = int(0.8 * len(df_pandas))
    val_size = int(0.1 * len(df_pandas))
    
    train_df = df_pandas[:train_size].copy()
    val_df = df_pandas[train_size:train_size + val_size].copy()
    test_df = df_pandas[train_size + val_size:].copy()
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples") 
    print(f"  Test: {len(test_df)} samples")
    
    # Create simple statistics for our features (normally done by PyTorch Frame)
    col_stats = train_dataset.col_stats
    
    # 4. Initialize multi-task model
    print(f"\nCreating multi-task model...")
    print(f"Using device: {device}")
    
    model = MultiTaskTabularModel(
        channels=64,
        num_transformer_layers=2,  # Reduced for faster testing
        num_heads=8,
        col_stats=col_stats,
        col_names_dict=actual_col_names_dict,
        target_config=target_config,
    ).to(device)
    
    print(f"\nModel architecture:")
    print(f"  Shared encoder: {sum(p.numel() for p in model.encoder.parameters())} parameters")
    print(f"  Transformer layers: {len(model.convs)} layers")
    print(f"  Output heads: {len(model.output_heads)} tasks")
    
    # Create TensorFrames for each split
    train_tf = create_multi_task_tensor_frame(train_df, actual_col_names_dict, target_config)
    val_tf = create_multi_task_tensor_frame(val_df, actual_col_names_dict, target_config)
    
    # Create data loaders
    train_loader = DataLoader(train_tf, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_tf, batch_size=512, shuffle=False)
    
    print(f"\nPreparing data loaders...")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # 6. Training loop with multi-task loss
    print("\n" + "="*60)
    print("MULTI-TASK TRAINING")
    print("="*60)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 3  # Reduced for faster testing
    
    # Track training history
    train_history = {task: [] for task in target_config.keys()}
    train_history['total'] = []
    val_history = {task: [] for task in target_config.keys()}
    val_history['total'] = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = {task: 0.0 for task in target_config.keys()}
        train_total_loss = 0.0
        train_batches = 0
        
        for batch_tf in train_loader:
            batch_tf = batch_tf.to(device)
            
            # Forward pass
            predictions = model(batch_tf)
            
            # Extract targets for each task
            targets = {}
            for i, (task_name, config) in enumerate(target_config.items()):
                targets[task_name] = batch_tf.y[:, i]
            
            # Compute losses for each task
            losses = compute_multi_task_loss(predictions, targets, target_config)
            
            # Combined loss (equal weighting for simplicity)
            total_loss = sum(losses.values())
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            for task_name, loss in losses.items():
                train_losses[task_name] += loss.item()
            train_total_loss += total_loss.item()
            train_batches += 1
        
        # Average training losses
        avg_train_losses = {task: loss / train_batches for task, loss in train_losses.items()}
        avg_train_total = train_total_loss / train_batches
        
        # Validation phase
        model.eval()
        val_losses = {task: 0.0 for task in target_config.keys()}
        val_total_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_tf in val_loader:
                batch_tf = batch_tf.to(device)
                predictions = model(batch_tf)
                
                # Extract targets
                targets = {}
                for i, (task_name, config) in enumerate(target_config.items()):
                    targets[task_name] = batch_tf.y[:, i]
                
                # Compute losses
                losses = compute_multi_task_loss(predictions, targets, target_config)
                total_loss = sum(losses.values())
                
                # Track losses
                for task_name, loss in losses.items():
                    val_losses[task_name] += loss.item()
                val_total_loss += total_loss.item()
                val_batches += 1
        
        # Average validation losses
        avg_val_losses = {task: loss / val_batches for task, loss in val_losses.items()}
        avg_val_total = val_total_loss / val_batches
        
        # Store history
        for task in target_config.keys():
            train_history[task].append(avg_train_losses[task])
            val_history[task].append(avg_val_losses[task])
        train_history['total'].append(avg_train_total)
        val_history['total'].append(avg_val_total)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1:2d}/{num_epochs}")
        print(f"  Total Loss    - Train: {avg_train_total:.4f}, Val: {avg_val_total:.4f}")
        
        for task_name, config in target_config.items():
            task_type = config['type']
            train_loss = avg_train_losses[task_name]
            val_loss = avg_val_losses[task_name]
            print(f"  {task_name:12s} - Train: {train_loss:.4f}, Val: {val_loss:.4f} [{task_type}]")
    
    # 7. Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    val_results = evaluate_multi_task(model, val_loader, target_config, device)
    
    print("Validation Results:")
    for task_name, metrics in val_results.items():
        config = target_config[task_name]
        print(f"\n{task_name} ({config['type']}):")
        for metric_name, value in metrics.items():
            if metric_name in ['accuracy']:
                print(f"  {metric_name}: {value:.2f}%")
            else:
                print(f"  {metric_name}: {value:.4f}")
    
    execution_time = time.time() - start_time
    return execution_time


# Run on CPU
cpu_time = run_demo('cpu')

# Run on GPU if available
gpu_time = None
if torch.cuda.is_available():
    # Clear CUDA cache to ensure fair comparison
    torch.cuda.empty_cache()
    gpu_time = run_demo('cuda')

# Print summary
print("\nPerformance Summary for {}:".format(os.path.basename(__file__)))
print(f"CPU execution time: {cpu_time:.2f} seconds")
if gpu_time:
    print(f"GPU execution time: {gpu_time:.2f} seconds")
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"GPU speedup: {speedup:.2f}x faster than CPU")
else:
    print("GPU not available for testing")
