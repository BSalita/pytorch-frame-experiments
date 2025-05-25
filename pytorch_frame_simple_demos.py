"""Simple PyTorch Frame demonstrations for binary classification, regression, and multi-class classification task types."""

import torch
import torch.nn.functional as F
from torch_frame import TensorFrame, stype
from torch_frame.data import DataLoader
from torch_frame.datasets import Yandex
from torch_frame.nn import StypeWiseFeatureEncoder, LinearEncoder, EmbeddingEncoder
from torch.nn import Linear, Module
import numpy as np
import pandas as pd


class SimpleModel(Module):
    """Simple model for demonstration purposes."""
    
    def __init__(self, channels, col_stats, col_names_dict, num_outputs, task_type):
        super().__init__()
        self.task_type = task_type
        
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict={
                stype.numerical: LinearEncoder(),
                stype.categorical: EmbeddingEncoder(),
            },
        )
        
        self.output_layer = Linear(channels, num_outputs)

    def forward(self, tf: TensorFrame):
        x, _ = self.encoder(tf)
        x_pooled = x.mean(dim=1)
        output = self.output_layer(x_pooled)
        
        if self.task_type in ['binary', 'regression']:
            output = output.squeeze(-1)
        
        return output


def create_simple_tensor_frame(df, col_names_dict, target_col, task_type):
    """Create TensorFrame for simple demonstration."""
    feat_dict = {}
    
    # Numerical features
    if stype.numerical in col_names_dict:
        feat_dict[stype.numerical] = torch.tensor(
            df[col_names_dict[stype.numerical]].values, dtype=torch.float32
        )
    
    # Categorical features
    if stype.categorical in col_names_dict:
        categorical_data = []
        for col in col_names_dict[stype.categorical]:
            unique_vals = sorted(df[col].unique())
            val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
            encoded = df[col].map(val_to_idx).values
            categorical_data.append(encoded)
        feat_dict[stype.categorical] = torch.tensor(
            np.column_stack(categorical_data), dtype=torch.long
        )
    
    # Target
    if task_type == 'regression':
        y = torch.tensor(df[target_col].values, dtype=torch.float32)
    else:  # binary or categorical
        if target_col == 'target_col':
            y = torch.tensor(df[target_col].values, dtype=torch.long)
        else:
            unique_vals = sorted(df[target_col].unique())
            val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
            encoded = df[target_col].map(val_to_idx).values
            y = torch.tensor(encoded, dtype=torch.long)
    
    return TensorFrame(feat_dict=feat_dict, y=y, col_names_dict=col_names_dict)


def load_and_prepare_data():
    """Load and prepare the base dataset."""
    dataset = Yandex(root="./data/adult", name="adult")
    dataset.materialize()
    df = dataset.df.sample(n=3000, random_state=42).copy()
    temp_dataset = dataset[:0.1]
    col_names_dict = temp_dataset.tensor_frame.col_names_dict
    return df, temp_dataset, col_names_dict


def split_data(df, train_ratio=0.8):
    """Split dataframe into train and validation sets."""
    train_size = int(train_ratio * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]
    return train_df, val_df


def create_data_loaders(train_df, val_df, col_names_dict, target_col, task_type, batch_size=128):
    """Create training and validation data loaders."""
    train_tf = create_simple_tensor_frame(train_df, col_names_dict, target_col, task_type)
    val_tf = create_simple_tensor_frame(val_df, col_names_dict, target_col, task_type)
    
    train_loader = DataLoader(train_tf, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tf, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def setup_model_and_optimizer(temp_dataset, col_names_dict, num_outputs, task_type, channels=32, lr=0.01):
    """Setup model and optimizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel(
        channels=channels,
        col_stats=temp_dataset.col_stats,
        col_names_dict=col_names_dict,
        num_outputs=num_outputs,
        task_type=task_type
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer, device


def train_epoch_binary(model, train_loader, optimizer, device):
    """Train one epoch for binary classification."""
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_tf in train_loader:
        batch_tf = batch_tf.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_tf)
        loss = F.binary_cross_entropy_with_logits(predictions, batch_tf.y.float())
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        predicted = (torch.sigmoid(predictions) > 0.5).float()
        train_correct += (predicted == batch_tf.y.float()).sum().item()
        train_total += batch_tf.y.size(0)
    
    return train_loss, train_correct, train_total


def validate_epoch_binary(model, val_loader, device):
    """Validate one epoch for binary classification."""
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_tf in val_loader:
            batch_tf = batch_tf.to(device)
            predictions = model(batch_tf)
            loss = F.binary_cross_entropy_with_logits(predictions, batch_tf.y.float())
            
            val_loss += loss.item()
            predicted = (torch.sigmoid(predictions) > 0.5).float()
            val_correct += (predicted == batch_tf.y.float()).sum().item()
            val_total += batch_tf.y.size(0)
    
    return val_loss, val_correct, val_total


def train_epoch_regression(model, train_loader, optimizer, device):
    """Train one epoch for regression."""
    model.train()
    train_loss = 0.0
    
    for batch_tf in train_loader:
        batch_tf = batch_tf.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_tf)
        loss = F.mse_loss(predictions, batch_tf.y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    return train_loss


def validate_epoch_regression(model, val_loader, device):
    """Validate one epoch for regression."""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_tf in val_loader:
            batch_tf = batch_tf.to(device)
            predictions = model(batch_tf)
            loss = F.mse_loss(predictions, batch_tf.y)
            val_loss += loss.item()
    
    return val_loss


def train_epoch_multiclass(model, train_loader, optimizer, device):
    """Train one epoch for multi-class classification."""
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_tf in train_loader:
        batch_tf = batch_tf.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_tf)
        loss = F.cross_entropy(predictions, batch_tf.y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(predictions, 1)
        train_correct += (predicted == batch_tf.y).sum().item()
        train_total += batch_tf.y.size(0)
    
    return train_loss, train_correct, train_total


def validate_epoch_multiclass(model, val_loader, device):
    """Validate one epoch for multi-class classification."""
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_tf in val_loader:
            batch_tf = batch_tf.to(device)
            predictions = model(batch_tf)
            loss = F.cross_entropy(predictions, batch_tf.y)
            
            val_loss += loss.item()
            _, predicted = torch.max(predictions, 1)
            val_correct += (predicted == batch_tf.y).sum().item()
            val_total += batch_tf.y.size(0)
    
    return val_loss, val_correct, val_total


def prepare_binary_data(df):
    """Prepare data for binary classification demo."""
    df['is_senior'] = (df['N_feature_0'] >= 50).astype(int)
    return df, 'is_senior'


def prepare_regression_data(df):
    """Prepare data for regression demo."""
    np.random.seed(42)
    noise = np.random.normal(0, 2, len(df))
    df['noisy_age'] = df['N_feature_0'] + noise
    return df, 'noisy_age'


def prepare_multiclass_data(df):
    """Prepare data for multi-class classification demo."""
    df['age_group'] = pd.cut(df['N_feature_0'], 
                            bins=[0, 30, 45, 60, 100], 
                            labels=['Young', 'Adult', 'Middle', 'Senior'])
    return df, 'age_group'


def demo_binary_classification():
    """Quick binary classification demo."""
    print("="*60)
    print("BINARY CLASSIFICATION DEMO")
    print("="*60)
    
    # Prepare data
    df, temp_dataset, col_names_dict = load_and_prepare_data()
    df, target_col = prepare_binary_data(df)
    train_df, val_df = split_data(df)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_df, val_df, col_names_dict, target_col, 'binary'
    )
    
    # Setup model
    model, optimizer, device = setup_model_and_optimizer(
        temp_dataset, col_names_dict, num_outputs=1, task_type='binary'
    )
    
    # Training
    print(f"Training binary classifier (senior citizen prediction)...")
    
    for epoch in range(5):
        train_loss, train_correct, train_total = train_epoch_binary(
            model, train_loader, optimizer, device
        )
        val_loss, val_correct, val_total = validate_epoch_binary(
            model, val_loader, device
        )
        
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/5 - Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    print(f"✅ Binary Classification Demo Complete!")
    print(f"   Final Validation Accuracy: {val_acc:.2f}%")
    return model


def demo_regression():
    """Quick regression demo."""
    print("\n" + "="*60)
    print("REGRESSION DEMO")
    print("="*60)
    
    # Prepare data
    df, temp_dataset, col_names_dict = load_and_prepare_data()
    df, target_col = prepare_regression_data(df)
    train_df, val_df = split_data(df)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_df, val_df, col_names_dict, target_col, 'regression'
    )
    
    # Setup model
    model, optimizer, device = setup_model_and_optimizer(
        temp_dataset, col_names_dict, num_outputs=1, task_type='regression'
    )
    
    # Training
    print(f"Training regression model (age prediction)...")
    
    for epoch in range(5):
        train_loss = train_epoch_regression(model, train_loader, optimizer, device)
        val_loss = validate_epoch_regression(model, val_loader, device)
        
        train_mse = train_loss / len(train_loader)
        val_mse = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/5 - Train MSE: {train_mse:.4f}, "
              f"Val MSE: {val_mse:.4f}, Val RMSE: {np.sqrt(val_mse):.4f}")
    
    print(f"✅ Regression Demo Complete!")
    print(f"   Final Validation RMSE: {np.sqrt(val_mse):.4f}")
    return model


def demo_multiclass_classification():
    """Quick multi-class classification demo."""
    print("\n" + "="*60)
    print("MULTI-CLASS CLASSIFICATION DEMO")
    print("="*60)
    
    # Prepare data
    df, temp_dataset, col_names_dict = load_and_prepare_data()
    df, target_col = prepare_multiclass_data(df)
    train_df, val_df = split_data(df)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_df, val_df, col_names_dict, target_col, 'categorical'
    )
    
    # Setup model
    num_classes = len(df[target_col].unique())
    model, optimizer, device = setup_model_and_optimizer(
        temp_dataset, col_names_dict, num_outputs=num_classes, task_type='categorical'
    )
    
    # Training
    print(f"Training multi-class classifier ({num_classes} age groups)...")
    
    for epoch in range(5):
        train_loss, train_correct, train_total = train_epoch_multiclass(
            model, train_loader, optimizer, device
        )
        val_loss, val_correct, val_total = validate_epoch_multiclass(
            model, val_loader, device
        )
        
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/5 - Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    print(f"✅ Multi-Class Classification Demo Complete!")
    print(f"   Final Validation Accuracy: {val_acc:.2f}%")
    return model


if __name__ == "__main__":
    print("🚀 PYTORCH FRAME SIMPLE DEMONSTRATIONS")
    print("Showcasing different task types with quick examples")
    
    # Run all demos
    binary_model = demo_binary_classification()
    regression_model = demo_regression()
    multiclass_model = demo_multiclass_classification()
    
    print(f"\n" + "="*60)
    print("🎯 ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("✅ Binary Classification: Senior citizen prediction")
    print("✅ Regression: Age prediction with noise")
    print("✅ Multi-Class: Age group classification (4 classes)")
    print("\n🚀 PyTorch Frame handles all tabular learning scenarios!") 