"""PyTorch Frame multi-task training demo with epochs and loss tracking."""

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
from torch.nn import Linear, Module, ModuleList
import numpy as np
import pandas as pd

print(f"Running program: {os.path.basename(__file__)}")

def create_multi_target_dataset(original_dataset, sample_size=5000):
    """Create additional targets from the Adult dataset for multi-task learning."""
    # Take a smaller sample for demo
    df = original_dataset.df.sample(n=sample_size, random_state=42).copy()
    
    # Target 1: Original categorical target (income >50K) - kept as is
    # Target 2: Boolean target - whether person is senior (age >= 50)
    df['is_senior'] = (df['N_feature_0'] >= 50).astype(int)
    
    # Target 3: Regression target - predict education level with noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, len(df))
    df['education_years'] = df['N_feature_2'] + noise
    
    # Target 4: Multi-class categorical - work sector based on education
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


# Simple multi-task model
class SimpleMultiTaskModel(Module):
    def __init__(self, channels, col_stats, col_names_dict, target_config):
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
        
        # Simple shared layer
        self.shared_layer = Linear(channels, channels)
        
        # Task-specific output heads
        self.output_heads = ModuleList()
        for task_name, config in target_config.items():
            if config['type'] == 'regression':
                head = Linear(channels, 1)
            elif config['type'] == 'binary':
                head = Linear(channels, 1)  # Binary classification needs 1 output (logit)
            else:
                head = Linear(channels, config['num_classes'])
            self.output_heads.append(head)

    def forward(self, tf: TensorFrame) -> dict:
        # Shared representation
        x, _ = self.encoder(tf)
        x_pooled = x.mean(dim=1)
        x_shared = torch.relu(self.shared_layer(x_pooled))
        
        # Task-specific predictions
        outputs = {}
        for i, (task_name, config) in enumerate(self.target_config.items()):
            pred = self.output_heads[i](x_shared)
            if config['type'] == 'regression':
                pred = pred.squeeze(-1)
            outputs[task_name] = pred
        return outputs


def compute_multi_task_loss(predictions, targets, target_config):
    losses = {}
    for task_name, pred in predictions.items():
        target = targets[task_name]
        config = target_config[task_name]
        
        if config['type'] == 'regression':
            losses[task_name] = F.mse_loss(pred, target.float())
        elif config['type'] == 'binary':
            losses[task_name] = F.binary_cross_entropy_with_logits(pred.squeeze(), target.float())
        else:  # categorical
            losses[task_name] = F.cross_entropy(pred, target.long())
    return losses


def create_simple_tensor_frame(df_split, col_names_dict, target_config):
    """Create a simplified TensorFrame."""
    feat_dict = {}
    
    # Numerical features
    if stype.numerical in col_names_dict:
        numerical_data = torch.tensor(
            df_split[col_names_dict[stype.numerical]].values, 
            dtype=torch.float32
        )
        feat_dict[stype.numerical] = numerical_data
    
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
    target_tensors = []
    for task_name, config in target_config.items():
        col = config['column']
        if config['type'] == 'regression':
            target_tensor = torch.tensor(df_split[col].values, dtype=torch.float32)
        elif config['type'] == 'binary':
            target_tensor = torch.tensor(df_split[col].values, dtype=torch.long)
        else:  # categorical
            if col == 'target_col':
                target_tensor = torch.tensor(df_split[col].values, dtype=torch.long)
            else:
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
    print(f"PyTorch Frame Multi-Task Learning Demo (Device: {device_name})")
    print("="*60)
    
    # Ensure device is valid
    device = torch.device(device_name)
    
    # Load and prepare data
    print("\n1. Loading dataset...")
    dataset = Yandex(root="./data/adult", name="adult")
    dataset.materialize()
    
    enhanced_df = create_multi_target_dataset(dataset, sample_size=5000)
    print(f"   Using {len(enhanced_df)} samples for demo")
    
    # Target configuration
    target_config = {
        'income_category': {'column': 'target_col', 'type': 'categorical', 'num_classes': 2},
        'is_senior': {'column': 'is_senior', 'type': 'binary', 'num_classes': 2},
        'education_years': {'column': 'education_years', 'type': 'regression', 'num_classes': 1},
        'work_sector': {'column': 'work_sector', 'type': 'categorical', 
                       'num_classes': len(enhanced_df['work_sector'].unique())}
    }
    
    print("\n2. Multi-task targets:")
    for name, config in target_config.items():
        print(f"   {name}: {config['type']} ({config['num_classes']} classes/dims)")
    
    # Data preparation
    train_dataset = dataset[:0.8]
    actual_col_names_dict = train_dataset.tensor_frame.col_names_dict
    
    feature_columns = (actual_col_names_dict[stype.numerical] + 
                      actual_col_names_dict[stype.categorical])
    target_columns = [cfg['column'] for cfg in target_config.values()]
    df_filtered = enhanced_df[feature_columns + target_columns].copy()
    
    # Splits
    train_size = int(0.8 * len(df_filtered))
    train_df = df_filtered[:train_size].copy()
    val_df = df_filtered[train_size:].copy()
    
    print(f"\n3. Data splits:")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Validation: {len(val_df)} samples")
    
    # Create model
    print(f"\n4. Creating model (device: {device})...")
    
    model = SimpleMultiTaskModel(
        channels=32,
        col_stats=train_dataset.col_stats,
        col_names_dict=actual_col_names_dict,
        target_config=target_config,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Data loaders
    train_tf = create_simple_tensor_frame(train_df, actual_col_names_dict, target_config)
    val_tf = create_simple_tensor_frame(val_df, actual_col_names_dict, target_config)
    
    train_loader = DataLoader(train_tf, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_tf, batch_size=256, shuffle=False)
    
    print(f"\n5. Data loaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Training
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 3  # Reduced for faster testing
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = {task: 0.0 for task in target_config.keys()}
        train_total = 0.0
        num_batches = 0
        
        for batch_tf in train_loader:
            batch_tf = batch_tf.to(device)
            predictions = model(batch_tf)
            
            targets = {}
            for i, (task_name, _) in enumerate(target_config.items()):
                targets[task_name] = batch_tf.y[:, i]
            
            losses = compute_multi_task_loss(predictions, targets, target_config)
            total_loss = sum(losses.values())
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            for task, loss in losses.items():
                train_losses[task] += loss.item()
            train_total += total_loss.item()
            num_batches += 1
        
        # Validation
        model.eval()
        val_losses = {task: 0.0 for task in target_config.keys()}
        val_total = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_tf in val_loader:
                batch_tf = batch_tf.to(device)
                predictions = model(batch_tf)
                
                targets = {}
                for i, (task_name, _) in enumerate(target_config.items()):
                    targets[task_name] = batch_tf.y[:, i]
                
                losses = compute_multi_task_loss(predictions, targets, target_config)
                total_loss = sum(losses.values())
                
                for task, loss in losses.items():
                    val_losses[task] += loss.item()
                val_total += total_loss.item()
                val_batches += 1
        
        # Print results
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Total Loss - Train: {train_total/num_batches:.4f}, Val: {val_total/val_batches:.4f}")
        
        for task_name, config in target_config.items():
            train_loss = train_losses[task_name] / num_batches
            val_loss = val_losses[task_name] / val_batches
            print(f"  {task_name:15s} - Train: {train_loss:.4f}, Val: {val_loss:.4f} [{config['type']}]")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    
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