# Making DataFrame Columns Discoverable by LLMs

This guide provides multiple approaches to make DataFrame column purposes and meanings discoverable by Large Language Models (LLMs), enabling better code understanding and automated analysis.

## 🎯 Problem Statement

When working with datasets, especially those with generic column names like `C_feature_0`, `N_feature_1`, etc., it's challenging for LLMs to understand:
- What each column represents
- The business meaning of features
- How columns should be processed
- Which columns are targets vs features
- Data quality considerations

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Approach 1: Rich Metadata Schema](#approach-1-rich-metadata-schema)
3. [Approach 2: Pandas Attrs Integration](#approach-2-pandas-attrs-integration)
4. [Approach 3: External Schema Files](#approach-3-external-schema-files)
5. [Approach 4: Auto-Discovery](#approach-4-auto-discovery)
6. [PyTorch Frame Integration](#pytorch-frame-integration)
7. [Best Practices](#best-practices)
8. [Examples](#examples)

## 🚀 Quick Start

```python
from pytorch_frame_metadata import TorchFrameColumnAnnotator

# Create annotator
annotator = TorchFrameColumnAnnotator()

# Annotate columns
annotator.annotate_column(
    column_name="age",
    description="Age of individual in years",
    torch_frame_stype="numerical",
    purpose="feature",
    business_meaning="Income typically increases with age/experience"
)

# Generate LLM-friendly description
print(annotator.describe_for_llm())
```

## 🏗️ Approach 1: Rich Metadata Schema

Create comprehensive metadata using dataclasses:

```python
@dataclass
class ColumnMetadata:
    name: str
    description: str
    data_type: str
    semantic_type: str  # 'categorical', 'numerical', 'text', etc.
    purpose: str        # 'target', 'feature', 'identifier', 'metadata'
    unit: Optional[str] = None
    valid_range: Optional[tuple] = None
    categories: Optional[List[str]] = None
    business_meaning: Optional[str] = None
```

**Benefits:**
- ✅ Comprehensive metadata
- ✅ Type safety
- ✅ Extensible structure
- ✅ JSON serializable

**Use when:** You need rich, structured metadata with strong typing

## 🔗 Approach 2: Pandas Attrs Integration

Store metadata directly in DataFrame attributes:

```python
class SmartDataFrame:
    def __init__(self, df: pd.DataFrame, schema: DatasetSchema):
        self.df = df
        self.schema = schema
        
        # Store in pandas attrs (discoverable by inspection)
        self.df.attrs['column_metadata'] = {
            name: asdict(meta) for name, meta in schema.columns.items()
        }
```

**Benefits:**
- ✅ Metadata travels with DataFrame
- ✅ Built-in pandas feature
- ✅ Accessible via `df.attrs`
- ✅ Preserved through operations

**Use when:** You want metadata to stay attached to DataFrames

## 📄 Approach 3: External Schema Files

Define schemas in YAML/JSON files:

```yaml
dataset:
  name: "Adult Income Dataset"
  description: "Predict income level from census data"
  
columns:
  age:
    description: "Age of individual in years"
    data_type: "int64"
    semantic_type: "numerical"
    purpose: "feature"
    business_meaning: "Income typically increases with age"
```

**Benefits:**
- ✅ Version controlled
- ✅ Language agnostic
- ✅ Easy to edit
- ✅ Shareable across teams

**Use when:** You need collaborative schema management

## 🤖 Approach 4: Auto-Discovery

Automatically infer column metadata:

```python
def analyze_column_automatically(series: pd.Series) -> ColumnMetadata:
    # Infer semantic type from data
    if series.dtype == 'object':
        unique_ratio = series.nunique() / len(series)
        semantic_type = 'categorical' if unique_ratio < 0.1 else 'text'
    elif series.dtype in ['int64', 'float64']:
        semantic_type = 'numerical'
    
    # Generate description
    description = f"Auto-detected {semantic_type} column"
    # ... more inference logic
```

**Benefits:**
- ✅ No manual work required
- ✅ Quick setup
- ✅ Good starting point
- ✅ Consistent format

**Use when:** You need quick metadata generation for exploration

## 🔥 PyTorch Frame Integration

Special integration for PyTorch Frame workflows:

```python
from pytorch_frame_metadata import create_yandex_adult_annotator

# Get pre-configured annotator
annotator = create_yandex_adult_annotator()

# Create TensorFrame with metadata
tensor_frame = annotator.create_tensor_frame(df)

# Get feature/target columns
features = annotator.get_feature_columns()
targets = annotator.get_target_columns()
stypes = annotator.get_stypes_dict()
```

**Features:**
- ✅ PyTorch Frame stype mapping
- ✅ Feature/target separation
- ✅ Business context annotations
- ✅ Preprocessing notes
- ✅ Fairness considerations

## 📝 Best Practices

### 1. **Use Descriptive Column Names When Possible**
```python
# ❌ Bad
df['f1'] = age_data

# ✅ Good  
df['age_years'] = age_data
```

### 2. **Include Business Context**
```python
annotator.annotate_column(
    'education_years',
    description="Years of education completed",
    business_meaning="Higher education strongly correlates with income potential",
    preprocessing_notes="Consider binning for non-linear relationships"
)
```

### 3. **Document Data Quality Issues**
```python
annotator.annotate_column(
    'income',
    description="Annual income in USD",
    preprocessing_notes="Contains outliers above $500K, may need capping"
)
```

### 4. **Mark Sensitive Attributes**
```python
annotator.annotate_column(
    'race',
    description="Racial category",
    preprocessing_notes="Protected attribute - monitor for bias in predictions"
)
```

### 5. **Provide Examples**
```python
annotator.annotate_column(
    'workclass',
    description="Type of employment",
    categories=['Private', 'Self-emp-not-inc', 'Federal-gov'],
    examples=['Private', 'Self-emp-inc', 'State-gov']
)
```

## 🎯 Use Cases

| Scenario | Recommended Approach |
|----------|---------------------|
| **Exploratory Analysis** | Auto-Discovery |
| **Production ML Pipeline** | Rich Metadata Schema |
| **Team Collaboration** | External Schema Files |
| **Research/Notebooks** | Pandas Attrs Integration |
| **PyTorch Frame Projects** | PyTorch Frame Integration |

## 🔧 Example Files

1. **`column_metadata_example.py`** - Complete metadata system implementation
2. **`pytorch_frame_metadata.py`** - PyTorch Frame specific utilities
3. **`discoverable_pytorch_frame_example.py`** - Full ML pipeline example

## 🚀 Getting Started

1. **Choose your approach** based on use case
2. **Start with auto-discovery** for quick setup
3. **Add business context** manually
4. **Export/save metadata** for reuse
5. **Generate LLM descriptions** for documentation

## 💡 Tips for LLM Discoverability

1. **Use natural language descriptions**
2. **Include business context and domain knowledge**
3. **Specify units and valid ranges**
4. **Document preprocessing requirements**
5. **Mark sensitive/protected attributes**
6. **Provide concrete examples**
7. **Explain relationships between features**

## 🔍 Debugging Column Issues

If you encounter column name mismatches:

```python
# Inspect actual dataset structure
def inspect_dataset(df):
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    print(f"Sample data:\n{df.head()}")
    
    for col in df.columns:
        print(f"{col}: {df[col].dtype}, {df[col].nunique()} unique")
```

## 📊 Integration Examples

### With Pandas Profiling
```python
# Generate profile with metadata
profile = ProfileReport(df, title="Dataset with Metadata")
profile.config.html.style.primary_color = "#337ab7"
```

### With MLflow
```python
# Log metadata as artifacts
mlflow.log_dict(annotator.column_info, "column_metadata.json")
```

### With DVC
```python
# Version control schemas
# dvc add schema.yaml
# git add schema.yaml.dv
```

This comprehensive approach ensures that your DataFrame columns are fully discoverable and understandable by both humans and LLMs, leading to better code quality, faster debugging, and more maintainable ML pipelines. 