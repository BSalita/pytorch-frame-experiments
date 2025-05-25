"""
PyTorch Frame Metadata Integration
Utility for making DataFrame columns discoverable in PyTorch Frame context
"""

import pandas as pd
import torch
from torch_frame import stype, TensorFrame
from torch_frame.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import json

class TorchFrameColumnAnnotator:
    """
    Annotates DataFrames with PyTorch Frame specific metadata
    Makes column purposes and types discoverable by LLMs
    """
    
    def __init__(self):
        self.column_info: Dict[str, Dict[str, Any]] = {}
        self.stype_mapping = {
            'categorical': stype.categorical,
            'numerical': stype.numerical, 
            'text_embedded': stype.text_embedded,
            'text_tokenized': stype.text_tokenized,
            'multicategorical': stype.multicategorical,
            'timestamp': stype.timestamp,
            'sequence_numerical': stype.sequence_numerical
        }
    
    def annotate_column(self, 
                       column_name: str,
                       description: str,
                       torch_frame_stype: str,
                       purpose: str = 'feature',
                       categories: Optional[List] = None,
                       preprocessing_notes: Optional[str] = None,
                       business_meaning: Optional[str] = None):
        """
        Annotate a column with comprehensive metadata
        
        Args:
            column_name: Name of the column
            description: Human-readable description
            torch_frame_stype: PyTorch Frame semantic type ('categorical', 'numerical', etc.)
            purpose: 'feature', 'target', 'identifier', 'metadata'
            categories: List of valid categories (for categorical columns)
            preprocessing_notes: Any special preprocessing requirements
            business_meaning: Business context and interpretation
        """
        
        self.column_info[column_name] = {
            'description': description,
            'torch_frame_stype': torch_frame_stype,
            'stype_object': self.stype_mapping[torch_frame_stype],
            'purpose': purpose,
            'categories': categories,
            'preprocessing_notes': preprocessing_notes,
            'business_meaning': business_meaning
        }
    
    def get_stypes_dict(self) -> Dict[str, Any]:
        """Get stypes dictionary for TensorFrame.from_df()"""
        feature_stypes = {}
        for col_name, info in self.column_info.items():
            if info['purpose'] == 'feature':
                feature_stypes[col_name] = info['stype_object']
        return feature_stypes
    
    def get_target_columns(self) -> List[str]:
        """Get list of target columns"""
        return [col for col, info in self.column_info.items() 
                if info['purpose'] == 'target']
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns"""
        return [col for col, info in self.column_info.items() 
                if info['purpose'] == 'feature']
    
    def describe_for_llm(self, df: Optional[pd.DataFrame] = None) -> str:
        """
        Generate comprehensive LLM-readable description
        
        Args:
            df: Optional DataFrame to include data statistics
        """
        
        description = "PyTorch Frame Dataset Column Metadata:\n\n"
        
        # Group by purpose
        purposes = {}
        for col_name, info in self.column_info.items():
            purpose = info['purpose']
            if purpose not in purposes:
                purposes[purpose] = []
            purposes[purpose].append((col_name, info))
        
        for purpose, columns in purposes.items():
            description += f"=== {purpose.upper()} COLUMNS ===\n"
            
            for col_name, info in columns:
                description += f"\n• {col_name}:\n"
                description += f"  Description: {info['description']}\n"
                description += f"  PyTorch Frame Type: {info['torch_frame_stype']}\n"
                
                if info['business_meaning']:
                    description += f"  Business Meaning: {info['business_meaning']}\n"
                
                if info['categories']:
                    description += f"  Categories: {info['categories']}\n"
                
                if info['preprocessing_notes']:
                    description += f"  Preprocessing: {info['preprocessing_notes']}\n"
                
                # Add data statistics if DataFrame provided
                if df is not None and col_name in df.columns:
                    series = df[col_name]
                    description += f"  Data Stats: {len(series)} rows, {series.isnull().sum()} missing"
                    
                    if info['torch_frame_stype'] == 'numerical':
                        description += f", range [{series.min():.2f}, {series.max():.2f}]"
                    elif info['torch_frame_stype'] == 'categorical':
                        description += f", {series.nunique()} unique values"
                    
                    description += "\n"
                
                description += "\n"
        
        return description
    
    def create_tensor_frame(self, df: pd.DataFrame, **kwargs) -> TensorFrame:
        """
        Create TensorFrame with proper column annotations
        
        Args:
            df: Input DataFrame
            **kwargs: Additional arguments for TensorFrame.from_df()
        """
        
        # Get feature columns and their stypes
        stypes = self.get_stypes_dict()
        target_cols = self.get_target_columns()
        
        # Create TensorFrame
        tensor_frame = TensorFrame.from_df(
            df=df,
            col_to_stype=stypes,
            target_col=target_cols,
            **kwargs
        )
        
        # Store metadata in tensor_frame for later discovery
        if hasattr(tensor_frame, 'attrs'):
            tensor_frame.attrs['column_metadata'] = self.column_info
        
        return tensor_frame
    
    def save_metadata(self, filepath: str):
        """Save metadata to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.column_info, f, indent=2, default=str)
    
    def load_metadata(self, filepath: str):
        """Load metadata from JSON file"""
        with open(filepath, 'r') as f:
            loaded_info = json.load(f)
        
        # Reconstruct stype objects
        for col_name, info in loaded_info.items():
            if 'torch_frame_stype' in info:
                stype_name = info['torch_frame_stype']
                if stype_name in self.stype_mapping:
                    info['stype_object'] = self.stype_mapping[stype_name]
        
        self.column_info = loaded_info

def create_yandex_adult_annotator() -> TorchFrameColumnAnnotator:
    """
    Create pre-configured annotator for Yandex Adult dataset
    Maps the generic column names to their actual meanings
    """
    
    annotator = TorchFrameColumnAnnotator()
    
    # Based on Adult dataset documentation, map generic names to meanings
    # Note: This mapping should be verified with actual dataset inspection
    
    # Categorical features (C_feature_*)
    annotator.annotate_column(
        'C_feature_0', 
        'Work class - type of employment (Private, Gov, Self-employed, etc.)',
        'categorical',
        purpose='feature',
        business_meaning='Employment sector affects income patterns'
    )
    
    annotator.annotate_column(
        'C_feature_1',
        'Education level - highest degree completed', 
        'categorical',
        purpose='feature',
        business_meaning='Higher education typically correlates with higher income'
    )
    
    annotator.annotate_column(
        'C_feature_2',
        'Marital status - relationship status',
        'categorical', 
        purpose='feature',
        business_meaning='Marital status affects household income dynamics'
    )
    
    annotator.annotate_column(
        'C_feature_3',
        'Occupation - specific job type',
        'categorical',
        purpose='feature', 
        business_meaning='Job role is primary determinant of income level'
    )
    
    annotator.annotate_column(
        'C_feature_4',
        'Relationship - family relationship role',
        'categorical',
        purpose='feature',
        business_meaning='Household role affects financial responsibility'
    )
    
    annotator.annotate_column(
        'C_feature_5',
        'Race - racial category',
        'categorical',
        purpose='feature',
        preprocessing_notes='Sensitive attribute - consider fairness implications'
    )
    
    annotator.annotate_column(
        'C_feature_6', 
        'Sex - gender',
        'categorical',
        purpose='feature',
        preprocessing_notes='Protected attribute - monitor for bias'
    )
    
    annotator.annotate_column(
        'C_feature_7',
        'Native country - country of birth',
        'categorical',
        purpose='feature',
        business_meaning='Immigration status may affect income opportunities'
    )
    
    # Numerical features (N_feature_*)
    annotator.annotate_column(
        'N_feature_0',
        'Age - age in years',
        'numerical',
        purpose='feature',
        business_meaning='Income typically increases with age/experience'
    )
    
    annotator.annotate_column(
        'N_feature_1', 
        'Final weight - sampling weight from census',
        'numerical',
        purpose='feature',
        preprocessing_notes='Statistical weight - may need normalization'
    )
    
    annotator.annotate_column(
        'N_feature_2',
        'Education number - numerical encoding of education level',
        'numerical', 
        purpose='feature',
        business_meaning='Quantified education level (1=min, 16=max)'
    )
    
    annotator.annotate_column(
        'N_feature_3',
        'Capital gain - income from asset sales',
        'numerical',
        purpose='feature',
        business_meaning='Investment income indicates wealth level'
    )
    
    annotator.annotate_column(
        'N_feature_4',
        'Capital loss - losses from asset sales', 
        'numerical',
        purpose='feature',
        business_meaning='Investment losses indicate financial activity'
    )
    
    annotator.annotate_column(
        'N_feature_5',
        'Hours per week - typical work hours',
        'numerical',
        purpose='feature', 
        business_meaning='Work hours directly affect total income'
    )
    
    # Target column
    annotator.annotate_column(
        'target_col',
        'Income level - binary classification target (>50K vs <=50K)',
        'categorical',
        purpose='target',
        categories=['<=50K', '>50K'],
        business_meaning='Primary prediction target for income classification'
    )
    
    return annotator

# Example usage function
def demo_pytorch_frame_metadata():
    """Demonstrate the metadata system with PyTorch Frame"""
    
    print("Creating Yandex Adult dataset annotator...")
    annotator = create_yandex_adult_annotator()
    
    print("\n" + "="*50)
    print("DATASET METADATA FOR LLM CONSUMPTION:")
    print("="*50)
    print(annotator.describe_for_llm())
    
    print("\nFeature columns:", annotator.get_feature_columns())
    print("Target columns:", annotator.get_target_columns())
    print("PyTorch Frame stypes:", list(annotator.get_stypes_dict().keys()))
    
    # Save metadata for future use
    annotator.save_metadata('yandex_adult_metadata.json')
    print("\nMetadata saved to 'yandex_adult_metadata.json'")

if __name__ == "__main__":
    demo_pytorch_frame_metadata() 