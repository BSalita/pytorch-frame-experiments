"""
Complete PyTorch Frame Example with Discoverable Column Metadata
Demonstrates how to use the metadata system in a real ML pipeline
"""

import pandas as pd
import torch
import torch.nn as nn
from torch_frame import stype, TensorFrame
from torch_frame.data import DataLoader
from torch_frame.datasets import Yandex
from torch_frame.nn import TabTransformer, FTTransformer
from pytorch_frame_metadata import TorchFrameColumnAnnotator, create_yandex_adult_annotator
import json

class DiscoverableTabularModel:
    """
    Tabular ML model with discoverable column metadata for LLM understanding
    """
    
    def __init__(self, annotator: TorchFrameColumnAnnotator):
        self.annotator = annotator
        self.model = None
        self.tensor_frame = None
        self.train_loader = None
        self.test_loader = None
        
    def prepare_data(self):
        """Load and prepare data with metadata"""
        print("Loading Yandex Adult dataset...")
        train_dataset = Yandex(root='data', name='adult', split='train')
        test_dataset = Yandex(root='data', name='adult', split='test')
        
        # Get the actual column mapping by inspecting the data
        print(f"Dataset columns: {train_dataset.df.columns.tolist()}")
        print(f"Dataset shape: {train_dataset.df.shape}")
        
        # Create TensorFrame using our annotated metadata
        print("\nCreating TensorFrame with metadata...")
        stypes = self.annotator.get_stypes_dict()
        target_cols = self.annotator.get_target_columns()
        
        # Filter stypes to only include columns that actually exist
        actual_columns = set(train_dataset.df.columns)
        filtered_stypes = {col: stype_obj for col, stype_obj in stypes.items() 
                          if col in actual_columns}
        filtered_targets = [col for col in target_cols if col in actual_columns]
        
        print(f"Using stypes for: {list(filtered_stypes.keys())}")
        print(f"Using targets: {filtered_targets}")
        
        # Create tensor frames
        self.train_tensor_frame = TensorFrame.from_df(
            df=train_dataset.df,
            col_to_stype=filtered_stypes,
            target_col=filtered_targets[0] if filtered_targets else None
        )
        
        self.test_tensor_frame = TensorFrame.from_df(
            df=test_dataset.df,
            col_to_stype=filtered_stypes,
            target_col=filtered_targets[0] if filtered_targets else None
        )
        
        # Store metadata in tensor frames
        self.train_tensor_frame.attrs = {'column_metadata': self.annotator.column_info}
        self.test_tensor_frame.attrs = {'column_metadata': self.annotator.column_info}
        
        # Create data loaders
        self.train_loader = DataLoader(self.train_tensor_frame, batch_size=128, shuffle=True)
        self.test_loader = DataLoader(self.test_tensor_frame, batch_size=128)
        
        print(f"Train TensorFrame shape: {self.train_tensor_frame.num_rows}")
        print(f"Test TensorFrame shape: {self.test_tensor_frame.num_rows}")
        
    def create_model(self):
        """Create model architecture"""
        print("\nCreating TabTransformer model...")
        
        # Get dimensions from tensor frame
        out_channels = self.train_tensor_frame.y.size(-1)
        
        self.model = TabTransformer(
            channels=64,
            num_layers=2,
            num_heads=4,
            col_stats=self.train_tensor_frame.col_stats,
            col_names_dict=self.train_tensor_frame.col_names_dict,
            out_channels=out_channels
        )
        
        print(f"Model created with {out_channels} output channels")
        
    def train_epoch(self, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            optimizer.zero_grad()
            
            pred = self.model(batch.tf)
            loss = criterion(pred, batch.tf.y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
        
    def evaluate(self):
        """Evaluate model on test set"""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                pred = self.model(batch.tf)
                pred_labels = torch.argmax(pred, dim=1)
                true_labels = batch.tf.y.squeeze()
                
                total_correct += (pred_labels == true_labels).sum().item()
                total_samples += true_labels.size(0)
                
        accuracy = total_correct / total_samples
        return accuracy
        
    def train_model(self, num_epochs=5):
        """Train the model"""
        if self.model is None:
            self.create_model()
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print(f"\nTraining for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(optimizer, criterion)
            accuracy = self.evaluate()
            
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Accuracy={accuracy:.4f}")
            
    def explain_model_for_llm(self) -> str:
        """Generate comprehensive explanation for LLM"""
        explanation = "DISCOVERABLE TABULAR ML MODEL EXPLANATION\n"
        explanation += "=" * 50 + "\n\n"
        
        # Dataset description
        explanation += self.annotator.describe_for_llm(self.train_tensor_frame.df)
        
        # Model architecture
        explanation += "\nMODEL ARCHITECTURE:\n"
        explanation += f"- Type: TabTransformer (Transformer-based tabular model)\n"
        explanation += f"- Feature columns: {len(self.annotator.get_feature_columns())}\n"
        explanation += f"- Target columns: {len(self.annotator.get_target_columns())}\n"
        explanation += f"- Model parameters: {sum(p.numel() for p in self.model.parameters()) if self.model else 'Not created'}\n"
        
        # Feature importance insights
        explanation += "\nFEATURE INSIGHTS:\n"
        for col_name, info in self.annotator.column_info.items():
            if info['purpose'] == 'feature' and info.get('business_meaning'):
                explanation += f"- {col_name}: {info['business_meaning']}\n"
                
        # Preprocessing notes
        explanation += "\nPREPROCESSING CONSIDERATIONS:\n"
        for col_name, info in self.annotator.column_info.items():
            if info.get('preprocessing_notes'):
                explanation += f"- {col_name}: {info['preprocessing_notes']}\n"
        
        return explanation

def inspect_actual_dataset():
    """Inspect the actual Yandex dataset to understand column mapping"""
    print("Inspecting actual Yandex Adult dataset structure...")
    
    train_dataset = Yandex(root='data', name='adult', split='train')
    df = train_dataset.df
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nColumn data types:")
    print(df.dtypes)
    
    print(f"\nSample values for each column:")
    for col in df.columns:
        unique_vals = df[col].unique()
        if len(unique_vals) <= 10:
            print(f"{col}: {unique_vals}")
        else:
            print(f"{col}: {unique_vals[:5]}... ({len(unique_vals)} unique values)")
            
    # Save inspection results
    inspection_results = {
        'columns': df.columns.tolist(),
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'sample_data': df.head().to_dict()
    }
    
    with open('dataset_inspection.json', 'w') as f:
        json.dump(inspection_results, f, indent=2, default=str)
    
    print(f"\nInspection results saved to 'dataset_inspection.json'")

def main():
    """Main function demonstrating discoverable tabular ML"""
    
    print("DISCOVERABLE PYTORCH FRAME EXAMPLE")
    print("=" * 40)
    
    # First, let's inspect the actual dataset
    inspect_actual_dataset()
    
    print("\n" + "=" * 40)
    print("CREATING ANNOTATED MODEL")
    print("=" * 40)
    
    # Create annotator with metadata
    annotator = create_yandex_adult_annotator()
    
    # Create discoverable model
    model = DiscoverableTabularModel(annotator)
    
    try:
        # Prepare data
        model.prepare_data()
        
        # Train model
        model.train_model(num_epochs=3)
        
        # Generate LLM explanation
        print("\n" + "=" * 50)
        print("LLM-DISCOVERABLE MODEL EXPLANATION")
        print("=" * 50)
        print(model.explain_model_for_llm())
        
    except Exception as e:
        print(f"Error during model training: {e}")
        print("This might be due to column name mismatches.")
        print("Check 'dataset_inspection.json' for actual column structure.")

if __name__ == "__main__":
    main() 