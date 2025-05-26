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
8. [Integration Examples](#integration-examples)
9. [Examples](#examples)

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

### Polars Alternative

**Current Status:** Polars does **not** have a built-in attrs feature like Pandas. However, there are workarounds:

```python
# Option 1: Custom namespace plugin (runtime only)
import polars as pl
from polars.api import register_dataframe_namespace

@register_dataframe_namespace("meta")
class MetaPlugin:
    def __init__(self, df): 
        self._df = df
    
    def set_metadata(self, **kwargs):
        # Store in global registry keyed by df id()
        pass

# Option 2: Subclassing approach  
class MetaDataFrame(pl.DataFrame):
    def __init__(self, data, metadata=None):
        super().__init__(data)
        self.metadata = metadata or {}

# Option 3: External metadata storage
metadata_registry = {}
def store_metadata(df, metadata):
    metadata_registry[id(df)] = metadata
```

**GitHub Issue:** There's an [active discussion](https://github.com/pola-rs/polars/issues/5117) about adding DataFrame-level metadata support to Polars. The community has proposed several workarounds, but no native solution exists yet.

**Recommendation:** For Polars, use external metadata storage or the PyTorch Frame integration approach below.

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

## 🔗 Related Resources

For comprehensive guidance on creating columns with metadata from scratch, see our dedicated guide:

📖 **[Creating Columns with Metadata: Best Practices Guide](CREATING_METADATA_COLUMNS.md)**

This guide covers:
- Schema-first design philosophy
- Column naming conventions and standards
- Industry frameworks (FAIR principles, Schema.org)
- Validation and quality frameworks
- Team collaboration guidelines
- Migration strategies from legacy schemas

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
[**pandas-profiling**](https://github.com/ydataai/ydata-profiling) (now **ydata-profiling**) is an automated EDA tool that generates comprehensive HTML reports from DataFrames.

```python
from ydata_profiling import ProfileReport

# Generate profile with metadata annotations
profile = ProfileReport(
    df, 
    title="Dataset with Rich Metadata",
    explorative=True,
    config_file="profiling_config.yaml"
)

# Add custom metadata to report
profile.config.html.style.primary_color = "#337ab7"
profile.config.html.navbar_show = True

# Include column descriptions in report
for col_name, meta in annotator.column_info.items():
    if col_name in df.columns:
        # Add business context to profiling
        profile.description_set[col_name] = {
            "description": meta['description'],
            "business_meaning": meta.get('business_meaning', ''),
            "preprocessing_notes": meta.get('preprocessing_notes', '')
        }

profile.to_file("enhanced_report.html")
```

**Benefits:**
- Rich statistical analysis with business context
- Automated data quality assessment
- Beautiful HTML reports with metadata annotations
- Correlation analysis with feature explanations

### With MLflow
[**MLflow**](https://mlflow.org/) is an open-source platform for managing ML lifecycle, including experimentation, reproducibility, and deployment.

```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Start MLflow run with metadata tracking
with mlflow.start_run(run_name="tabular_model_with_metadata"):
    
    # Log dataset metadata as artifacts
    mlflow.log_dict(annotator.column_info, "column_metadata.json")
    
    # Log dataset schema
    schema_info = {
        'dataset_name': schema.name,
        'num_features': len(annotator.get_feature_columns()),
        'num_targets': len(annotator.get_target_columns()),
        'feature_types': {col: info['torch_frame_stype'] 
                         for col, info in annotator.column_info.items()},
        'business_context': {col: info.get('business_meaning', '') 
                           for col, info in annotator.column_info.items()}
    }
    mlflow.log_dict(schema_info, "dataset_schema.json")
    
    # Log preprocessing notes for reproducibility
    preprocessing_notes = {
        col: info.get('preprocessing_notes', '') 
        for col, info in annotator.column_info.items() 
        if info.get('preprocessing_notes')
    }
    mlflow.log_dict(preprocessing_notes, "preprocessing_requirements.json")
    
    # Train model and log with metadata context
    model = train_your_model(df, annotator)
    mlflow.sklearn.log_model(
        model, 
        "model",
        metadata={
            "feature_importance_context": annotator.describe_for_llm(),
            "data_schema_version": "1.0"
        }
    )
    
    # Log feature importance with business context
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(
            annotator.get_feature_columns(), 
            model.feature_importances_
        ))
        mlflow.log_dict(feature_importance, "feature_importance.json")
```

**Benefits:**
- Complete model lifecycle tracking with business context
- Reproducible experiments with metadata versioning
- Easy model comparison with feature explanations
- Collaboration through shared understanding of data

### With DVC
[**DVC (Data Version Control)**](https://dvc.org/) is a version control system for ML projects, handling data, models, and pipelines.

```python
# dvc.yaml - Define pipeline with metadata dependencies
import yaml

pipeline_config = {
    'stages': {
        'validate_schema': {
            'cmd': 'python validate_metadata.py',
            'deps': ['schema.yaml', 'raw_data.csv'],
            'outs': ['validated_metadata.json'],
            'desc': 'Validate dataset against business schema'
        },
        'prepare_data': {
            'cmd': 'python prepare_data.py',
            'deps': ['validated_metadata.json', 'raw_data.csv'],
            'outs': ['processed_data.csv', 'column_transformations.json'],
            'desc': 'Apply transformations based on metadata'
        },
        'train_model': {
            'cmd': 'python train.py',
            'deps': ['processed_data.csv', 'column_transformations.json'],
            'outs': ['model.pkl', 'feature_importance.json'],
            'metrics': ['metrics.json'],
            'desc': 'Train model with metadata-aware features'
        }
    }
}

with open('dvc.yaml', 'w') as f:
    yaml.dump(pipeline_config, f)

# schema.yaml - Version controlled metadata schema
schema_file = {
    'dataset': {
        'name': 'Adult Income Classification',
        'version': '2.1.0',
        'description': 'Census data with rich business metadata'
    },
    'columns': annotator.column_info,
    'validation_rules': {
        'age': {'min': 16, 'max': 100},
        'income': {'categories': ['<=50K', '>50K']},
        'education_years': {'min': 1, 'max': 21}
    },
    'data_quality': {
        'max_missing_percentage': 0.05,
        'outlier_detection': True,
        'bias_monitoring': ['gender', 'race']
    }
}

with open('schema.yaml', 'w') as f:
    yaml.dump(schema_file, f)

# Track schema with DVC
# dvc add schema.yaml
# git add schema.yaml.dvc
# git commit -m "Add business metadata schema v2.1.0"

# params.yaml - Parameters with metadata context
params = {
    'model': {
        'algorithm': 'TabTransformer',
        'embedding_dim': 64,
        'num_attention_heads': 8
    },
    'features': {
        'categorical_features': annotator.get_categorical_features(),
        'numerical_features': annotator.get_numerical_features(),
        'sensitive_attributes': ['gender', 'race'],  # From metadata
        'high_importance_features': ['education', 'occupation']  # From business logic
    },
    'training': {
        'epochs': 50,
        'batch_size': 128,
        'fairness_constraints': True  # Enabled due to sensitive attributes
    }
}
```

**DVC Workflow Commands:**
```bash
# Initialize DVC in your project
dvc init

# Track large data files
dvc add raw_data.csv
git add raw_data.csv.dvc .gitignore

# Run pipeline with metadata validation
dvc repro

# Compare experiments with metadata context
dvc exp show --include-params features.categorical_features

# Share data with metadata
dvc push
git push
```

**Benefits:**
- Version control for data schemas and transformations
- Reproducible pipelines with metadata dependencies  
- Collaboration through shared data understanding
- Experiment tracking with business context
- Data lineage with transformation reasoning

This comprehensive approach ensures that your DataFrame columns are fully discoverable and understandable by both humans and LLMs, leading to better code quality, faster debugging, and more maintainable ML pipelines. 