"""
Column Metadata Discovery Examples for LLM Understanding
Demonstrates various approaches to make DataFrame column purposes discoverable
"""

import pandas as pd
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Literal
from enum import Enum

# ==================== Approach 1: Dataclass Schema ====================

@dataclass
class ColumnMetadata:
    """Rich metadata for a DataFrame column"""
    name: str
    description: str
    data_type: str
    semantic_type: str  # 'categorical', 'numerical', 'text', 'datetime', 'binary'
    purpose: str  # 'target', 'feature', 'identifier', 'metadata'
    unit: Optional[str] = None
    valid_range: Optional[tuple] = None
    categories: Optional[List[str]] = None
    missing_allowed: bool = True
    examples: Optional[List[str]] = None

class DatasetSchema:
    """Complete schema for a dataset with discoverable metadata"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.columns: Dict[str, ColumnMetadata] = {}
        
    def add_column(self, metadata: ColumnMetadata):
        """Add column metadata"""
        self.columns[metadata.name] = metadata
        
    def describe_for_llm(self) -> str:
        """Generate LLM-friendly description of the dataset"""
        description = f"Dataset: {self.name}\n{self.description}\n\nColumns:\n"
        
        for col_name, meta in self.columns.items():
            description += f"\n- {col_name}: {meta.description}"
            description += f"\n  Type: {meta.data_type} ({meta.semantic_type})"
            description += f"\n  Purpose: {meta.purpose}"
            
            if meta.unit:
                description += f"\n  Unit: {meta.unit}"
            if meta.valid_range:
                description += f"\n  Range: {meta.valid_range}"
            if meta.categories:
                description += f"\n  Categories: {meta.categories}"
            if meta.examples:
                description += f"\n  Examples: {meta.examples}"
            description += "\n"
            
        return description
    
    def to_json(self) -> str:
        """Export schema as JSON for external storage"""
        schema_dict = {
            'name': self.name,
            'description': self.description,
            'columns': {name: asdict(meta) for name, meta in self.columns.items()}
        }
        return json.dumps(schema_dict, indent=2)

# ==================== Approach 2: Pandas Attrs Integration ====================

class SmartDataFrame:
    """DataFrame wrapper with rich metadata"""
    
    def __init__(self, df: pd.DataFrame, schema: DatasetSchema):
        self.df = df
        self.schema = schema
        
        # Store metadata in pandas attrs (discoverable by inspection)
        self.df.attrs['dataset_name'] = schema.name
        self.df.attrs['dataset_description'] = schema.description
        self.df.attrs['column_metadata'] = {
            name: asdict(meta) for name, meta in schema.columns.items()
        }
        
    def describe_column(self, column_name: str) -> str:
        """Get human-readable description of a column"""
        if column_name not in self.schema.columns:
            return f"Column '{column_name}' not found in schema"
            
        meta = self.schema.columns[column_name]
        return f"{column_name}: {meta.description} ({meta.semantic_type}, {meta.purpose})"
    
    def get_feature_columns(self) -> List[str]:
        """Get all feature columns for ML"""
        return [name for name, meta in self.schema.columns.items() 
                if meta.purpose == 'feature']
    
    def get_target_columns(self) -> List[str]:
        """Get all target columns for ML"""
        return [name for name, meta in self.schema.columns.items() 
                if meta.purpose == 'target']
    
    def llm_summary(self) -> str:
        """Generate comprehensive summary for LLM consumption"""
        summary = self.schema.describe_for_llm()
        summary += f"\nDataFrame Shape: {self.df.shape}"
        summary += f"\nMissing Values: {self.df.isnull().sum().to_dict()}"
        return summary

# ==================== Approach 3: YAML Configuration ====================

def create_yaml_schema_example():
    """Example of external YAML schema file"""
    yaml_content = """
dataset:
  name: "Adult Income Dataset"
  description: "Predict whether income exceeds $50K/yr based on census data"
  
columns:
  age:
    description: "Age of the individual in years"
    data_type: "int64"
    semantic_type: "numerical"
    purpose: "feature"
    unit: "years"
    valid_range: [17, 90]
    examples: ["39", "50", "25"]
    
  workclass:
    description: "Type of employment (government, private, self-employed, etc.)"
    data_type: "object"
    semantic_type: "categorical"
    purpose: "feature"
    categories: ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay"]
    
  education:
    description: "Highest level of education completed"
    data_type: "object" 
    semantic_type: "categorical"
    purpose: "feature"
    categories: ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm"]
    
  education_num:
    description: "Numerical representation of education level"
    data_type: "int64"
    semantic_type: "numerical" 
    purpose: "feature"
    unit: "education_years"
    valid_range: [1, 16]
    
  income:
    description: "Income level - target variable for classification"
    data_type: "object"
    semantic_type: "binary"
    purpose: "target"
    categories: ["<=50K", ">50K"]
"""
    return yaml_content

# ==================== Approach 4: Auto-Discovery Functions ====================

def analyze_column_automatically(series: pd.Series) -> ColumnMetadata:
    """Automatically infer column metadata from data"""
    name = series.name
    data_type = str(series.dtype)
    
    # Infer semantic type
    if series.dtype == 'object':
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.1:  # Low cardinality
            semantic_type = 'categorical'
            categories = series.unique().tolist()
        else:
            semantic_type = 'text'
            categories = None
    elif series.dtype in ['int64', 'float64']:
        semantic_type = 'numerical'
        categories = None
    elif series.dtype == 'bool':
        semantic_type = 'binary'
        categories = None
    else:
        semantic_type = 'unknown'
        categories = None
    
    # Infer purpose (basic heuristics)
    purpose = 'feature'  # Default
    if 'target' in name.lower() or 'label' in name.lower():
        purpose = 'target'
    elif 'id' in name.lower():
        purpose = 'identifier'
    
    # Get valid range for numerical data
    valid_range = None
    if semantic_type == 'numerical':
        valid_range = (series.min(), series.max())
    
    # Generate description
    description = f"Auto-detected {semantic_type} column"
    if semantic_type == 'categorical' and categories:
        description += f" with {len(categories)} categories"
    elif semantic_type == 'numerical':
        description += f" ranging from {valid_range[0]} to {valid_range[1]}"
    
    examples = series.dropna().astype(str).head(3).tolist()
    
    return ColumnMetadata(
        name=name,
        description=description,
        data_type=data_type,
        semantic_type=semantic_type,
        purpose=purpose,
        categories=categories,
        valid_range=valid_range,
        examples=examples
    )

def create_auto_schema(df: pd.DataFrame, dataset_name: str, dataset_description: str) -> DatasetSchema:
    """Automatically create schema from DataFrame"""
    schema = DatasetSchema(dataset_name, dataset_description)
    
    for column in df.columns:
        metadata = analyze_column_automatically(df[column])
        schema.add_column(metadata)
    
    return schema

# ==================== Example Usage ====================

def demo_discoverable_columns():
    """Demonstrate all approaches with sample data"""
    
    # Create sample data
    data = {
        'age': [39, 50, 38, 53, 28],
        'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Private', 'Private'],
        'education': ['Bachelors', 'Bachelors', 'HS-grad', '11th', 'Bachelors'],
        'education_num': [13, 13, 9, 7, 13],
        'income': ['<=50K', '<=50K', '<=50K', '<=50K', '<=50K']
    }
    df = pd.DataFrame(data)
    
    print("=== Manual Schema Creation ===")
    # Manual schema creation
    schema = DatasetSchema("Adult Income Sample", "Sample of adult income classification data")
    
    schema.add_column(ColumnMetadata(
        name="age",
        description="Age of individual in years",
        data_type="int64",
        semantic_type="numerical",
        purpose="feature",
        unit="years",
        valid_range=(17, 90),
        examples=["39", "50", "38"]
    ))
    
    schema.add_column(ColumnMetadata(
        name="income",
        description="Income level - binary classification target",
        data_type="object", 
        semantic_type="binary",
        purpose="target",
        categories=["<=50K", ">50K"]
    ))
    
    # Create smart dataframe
    smart_df = SmartDataFrame(df, schema)
    print(smart_df.llm_summary())
    
    print("\n=== Auto-Generated Schema ===")
    # Auto-generated schema
    auto_schema = create_auto_schema(df, "Auto-Detected Dataset", "Automatically analyzed dataset")
    auto_smart_df = SmartDataFrame(df, auto_schema)
    print(auto_smart_df.llm_summary())

if __name__ == "__main__":
    demo_discoverable_columns() 