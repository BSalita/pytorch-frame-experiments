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
from dataclasses import dataclass

# Centralized Hyperparameters
@dataclass
class Hyperparameters:
    batch_size: int = 128
    channels: int = 32
    lr: float = 0.01
    epochs: int = 5
    train_ratio: float = 0.8
    sample_size: int = 3000
    random_state: int = 42
    yandex_root: str = "./data/adult"
    yandex_name: str = "adult"

CONFIG = Hyperparameters()


class SimpleModel(Module):
    """Simple model for demonstration purposes."""
    
    def __init__(self, channels, col_stats, col_names_dict, num_outputs, task_type):
        super().__init__()
        self.task_type = task_type
        
        # Ensure col_stats is not None and has content
        if col_stats is None:
            raise ValueError("col_stats cannot be None for SimpleModel initialization.")

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


def create_simple_tensor_frame(df, col_names_dict, col_stats, target_col, task_type, target_mapping=None):
    """Create TensorFrame for simple demonstration, using provided col_stats for consistency."""
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
            # Use the mapping from the global col_stats
            if col_stats is None or col not in col_stats:
                raise ValueError(f"Missing col_stats for categorical column: {col}")
            
            # The mapping is an OrderedSet or similar structure in torch-frame col_stats
            # Accessing it via .stats['mapping'] which is an OrderedSet
            category_mapping = col_stats[col].stats['mapping']
            
            # Map values using the get_index method of the OrderedSet,
            # which handles unknown values by assigning them to a specific index (often the last one).
            encoded = df[col].apply(lambda x: category_mapping.get_index(x)).values
            categorical_data.append(encoded)
        
        if categorical_data: # Ensure there's data before stacking
            feat_dict[stype.categorical] = torch.tensor(
                np.column_stack(categorical_data), dtype=torch.long
            )
        else:
            # Handle cases where there are categorical columns in col_names_dict but no data for them
            # This might mean an empty DataFrame or specific columns being empty.
            # Create an empty tensor of appropriate shape if needed or handle as error.
            # For now, we'll assume categorical_data will be populated if stype.categorical is present.
            pass 
            
    # Target
    if task_type == 'regression':
        y = torch.tensor(df[target_col].values, dtype=torch.float32)
    else:  # binary or categorical
        if target_mapping:
            # Apply the provided mapping (derived from training data)
            encoded = df[target_col].map(target_mapping).values
            # Check for NaNs which indicate values in val_df not present in train_df's target_mapping
            if np.isnan(encoded).any():
                # Option 1: Raise error
                # raise ValueError(f"Target column '{target_col}' in validation set contains values not seen in training set.")
                # Option 2: Fill with a specific value (e.g., -1 or a new category index if model handles it)
                # For now, let's be strict if a mapping is given. Or ensure mapping handles all.
                # A simple way if using pandas categories is to ensure all categories are known.
                # Or, ensure the mapping is comprehensive or has a default.
                # For this demo, we'll assume target_mapping from train_df is exhaustive for val_df targets.
                # If not, pd.map will produce NaNs. We should handle this.
                # Let's assume the mapping must be complete.
                if df[target_col].nunique() > len(target_mapping): # A simple check
                     print(f"Warning: Potential unseen target values in {target_col} for df split.")
            
            # Fill NaNs that may arise if a value in df[target_col] is not in target_mapping.
            # This is crucial for validation set if it has new target labels not in training.
            # A robust way is to ensure target_mapping includes all possible labels or has a default for unknowns.
            # For simplicity in this demo, we assume labels are consistent or training captured all.
            # If mapping results in NaN, it will cause issues with torch.tensor.
            # Let's convert to float first to keep NaNs, then decide how to handle.
            encoded_float = df[target_col].map(target_mapping).astype(float).values
            if np.isnan(encoded_float).any():
                # This path should ideally not be hit if target_mapping is from train_df and applied to val_df,
                # AND if all val_df targets were seen in train_df.
                # If new target classes appear in val, this is an issue.
                # For now, we'll assume this doesn't happen in this controlled demo setup.
                # If it could, one might map NaNs to a specific index (e.g., -1) or filter them out.
                print(f"Warning: Target column '{target_col}' produced NaNs after mapping. Ensure all target values are in the mapping.")
                # A simple fix: map NaNs to a value like -1, if the loss function can ignore it or it's handled.
                # Or, better, ensure the mapping is exhaustive from the start.
                # For this demo, we will proceed, but this is a point of fragility.
            
            y = torch.tensor(encoded, dtype=torch.long)
        else:
            # If no target_mapping is provided (e.g. for regression, or if target is already numerical)
            # For classification, if target is strings, this path needs mapping logic (usually done in create_data_loaders now)
            # This path is now less likely for classification as mapping is handled upstream.
            if df[target_col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[target_col].dtype):
                 # This should not happen if target_mapping is correctly passed for categorical targets
                raise ValueError("Categorical target column requires a target_mapping.")
            y = torch.tensor(df[target_col].values, dtype=torch.long)
    
    return TensorFrame(feat_dict=feat_dict, y=y, col_names_dict=col_names_dict)


def load_and_prepare_data():
    """Load the base dataset and return df, col_stats, and col_names_dict."""
    dataset = Yandex(root=CONFIG.yandex_root, name=CONFIG.yandex_name)
    dataset.materialize()
    # Sample AFTER materializing to ensure col_stats are from the full dataset
    df_sampled = dataset.df.sample(n=CONFIG.sample_size, random_state=CONFIG.random_state).copy()
    
    # Use col_stats and col_names_dict from the full dataset
    # These are generated by torch-frame during materialize based on the whole dataset
    return df_sampled, dataset.col_stats, dataset.tensor_frame.col_names_dict


def split_data(df, train_ratio=CONFIG.train_ratio):
    """Split dataframe into train and validation sets."""
    train_size = int(train_ratio * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]
    return train_df, val_df


def create_data_loaders(train_df, val_df, col_names_dict, col_stats, target_col, task_type, batch_size=CONFIG.batch_size):
    """Create training and validation data loaders with consistent target encoding."""
    
    train_target_mapping = None
    num_classes = None

    if task_type in ['binary', 'categorical']:
        # Ensure target column exists
        if target_col not in train_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in training dataframe.")

        # Create mapping from unique values in the training set's target column
        # This ensures validation set uses the same mapping.
        # Convert to list then sort for consistent mapping, especially if target is string.
        unique_target_values = sorted(list(train_df[target_col].unique()))
        train_target_mapping = {val: idx for idx, val in enumerate(unique_target_values)}
        num_classes = len(train_target_mapping)

        if task_type == 'binary' and num_classes != 2:
            # If user designated binary, but target has != 2 classes after mapping from train_df
            print(f"Warning: Binary task type specified, but training target '{target_col}' has {num_classes} unique values: {unique_target_values}. Proceeding, but check data preparation.")
            # For a true binary case, this check might be stricter.
            # Here, we allow it but warn. The loss function (BCEWithLogits) expects single output.


    train_tf = create_simple_tensor_frame(train_df, col_names_dict, col_stats, target_col, task_type, target_mapping=train_target_mapping)
    val_tf = create_simple_tensor_frame(val_df, col_names_dict, col_stats, target_col, task_type, target_mapping=train_target_mapping)
    
    train_loader = DataLoader(train_tf, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tf, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, num_classes # Return num_classes for model setup


def setup_model_and_optimizer(col_stats, col_names_dict, num_outputs, task_type, channels=CONFIG.channels, lr=CONFIG.lr):
    """Setup model and optimizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel(
        channels=channels,
        col_stats=col_stats, # Pass the globally derived col_stats
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
    # Ensure 'N_feature_0' is numeric for comparison
    df['N_feature_0'] = pd.to_numeric(df['N_feature_0'], errors='coerce')
    df.dropna(subset=['N_feature_0'], inplace=True) # Drop rows where conversion failed
    df['is_senior'] = (df['N_feature_0'] >= 50).astype(int) # Target is already 0 or 1
    return df, 'is_senior'


def prepare_regression_data(df):
    """Prepare data for regression demo."""
    # Ensure 'N_feature_0' is numeric
    df['N_feature_0'] = pd.to_numeric(df['N_feature_0'], errors='coerce')
    df.dropna(subset=['N_feature_0'], inplace=True)
    np.random.seed(CONFIG.random_state)
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
    df, col_stats, col_names_dict = load_and_prepare_data()
    df, target_col = prepare_binary_data(df)
    train_df, val_df = split_data(df)
    
    # Create data loaders
    train_loader, val_loader, num_classes = create_data_loaders(
        train_df, val_df, col_names_dict, col_stats, target_col, 'binary'
    )
    
    # Setup model
    model, optimizer, device = setup_model_and_optimizer(
        col_stats, col_names_dict, num_outputs=1, task_type='binary'
    )
    
    # Training
    print(f"Training binary classifier (senior citizen prediction) for {CONFIG.epochs} epochs...")
    
    final_val_acc = 0.0
    for epoch in range(CONFIG.epochs):
        train_loss, train_correct, train_total_samples = train_epoch_binary( # train_total_samples not used for avg loss here
            model, train_loader, optimizer, device
        )
        # train_epoch_binary returns total loss, total correct, total samples
        # We need average loss per batch for printing, or per sample
        avg_train_loss = train_loss / len(train_loader) # Avg loss per batch

        val_loss, val_correct, val_total_samples = validate_epoch_binary(
            model, val_loader, device
        )
        avg_val_loss = val_loss / len(val_loader) # Avg loss per batch
        
        train_acc = 100.0 * train_correct / train_total_samples if train_total_samples > 0 else 0
        val_acc = 100.0 * val_correct / val_total_samples if val_total_samples > 0 else 0
        final_val_acc = val_acc
        
        print(f"Epoch {epoch+1}/{CONFIG.epochs} - Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    print(f"✅ Binary Classification Demo Complete!")
    print(f"   Final Validation Accuracy: {final_val_acc:.2f}%")
    return model


def demo_regression():
    """Quick regression demo."""
    print("\n" + "="*60)
    print("REGRESSION DEMO")
    print("="*60)
    
    # Prepare data
    df, col_stats, col_names_dict = load_and_prepare_data()
    df, target_col = prepare_regression_data(df)
    train_df, val_df = split_data(df)
    
    # Create data loaders
    # For regression, num_classes is not strictly needed from create_data_loaders but returned for consistency
    train_loader, val_loader, _ = create_data_loaders(
        train_df, val_df, col_names_dict, col_stats, target_col, 'regression'
    )
    
    # Setup model (num_outputs=1 for regression)
    model, optimizer, device = setup_model_and_optimizer(
        col_stats, col_names_dict, num_outputs=1, task_type='regression'
    )
    
    # Training
    print(f"Training regression model (age prediction) for {CONFIG.epochs} epochs...")
    
    final_val_rmse = 0.0
    for epoch in range(CONFIG.epochs):
        train_loss_sum = train_epoch_regression(model, train_loader, optimizer, device) # returns sum of losses
        val_loss_sum = validate_epoch_regression(model, val_loader, device) # returns sum of losses
        
        # train_epoch_regression and validate_epoch_regression now return avg loss per sample
        train_mse = train_loss_sum 
        val_mse = val_loss_sum
        final_val_rmse = np.sqrt(val_mse) if val_mse >= 0 else float('nan')
        
        print(f"Epoch {epoch+1}/{CONFIG.epochs} - Train MSE: {train_mse:.4f}, "
              f"Val MSE: {val_mse:.4f}, Val RMSE: {final_val_rmse:.4f}")
    
    print(f"✅ Regression Demo Complete!")
    print(f"   Final Validation RMSE: {final_val_rmse:.4f}")
    return model


def demo_multiclass_classification():
    """Quick multi-class classification demo."""
    print("\n" + "="*60)
    print("MULTI-CLASS CLASSIFICATION DEMO")
    print("="*60)
    
    # Prepare data
    df, col_stats, col_names_dict = load_and_prepare_data()
    df, target_col = prepare_multiclass_data(df) # target_col is 'age_group' with string labels
    train_df, val_df = split_data(df)
    
    # Create data loaders
    # num_classes will be determined from train_df's target column
    train_loader, val_loader, num_classes = create_data_loaders(
        train_df, val_df, col_names_dict, col_stats, target_col, 'categorical'
    )
    
    # Setup model
    # num_classes is now correctly derived from the actual training target labels
    if num_classes is None or num_classes <= 1:
        raise ValueError(f"Multi-class task expects more than 1 class, but got {num_classes} from target '{target_col}'. Check data prep.")

    model, optimizer, device = setup_model_and_optimizer(
        col_stats, col_names_dict, num_outputs=num_classes, task_type='categorical'
    )
    
    # Training
    print(f"Training multi-class classifier ({num_classes} age groups) for {CONFIG.epochs} epochs...")
    
    final_val_acc = 0.0
    for epoch in range(CONFIG.epochs):
        # train_epoch_multiclass returns avg_loss, total_correct, total_samples
        train_avg_loss, train_correct, train_total_samples = train_epoch_multiclass(
            model, train_loader, optimizer, device
        )
        val_avg_loss, val_correct, val_total_samples = validate_epoch_multiclass(
            model, val_loader, device
        )
        
        train_acc = 100.0 * train_correct / train_total_samples if train_total_samples > 0 else 0
        val_acc = 100.0 * val_correct / val_total_samples if val_total_samples > 0 else 0
        final_val_acc = val_acc
        
        print(f"Epoch {epoch+1}/{CONFIG.epochs} - Train Loss: {train_avg_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Loss: {val_avg_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    print(f"✅ Multi-Class Classification Demo Complete!")
    print(f"   Final Validation Accuracy: {final_val_acc:.2f}%")
    return model


if __name__ == "__main__":
    print(f"🚀 PYTORCH FRAME SIMPLE DEMONSTRATIONS (Config: {CONFIG})")
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