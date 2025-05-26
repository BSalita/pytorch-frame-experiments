# 🧠 Natural Language to SQL Translation with Intelligent Column Prediction

A sophisticated system that leverages our **column metadata framework** to translate natural language queries into SQL with intelligent prediction of useful columns beyond the obvious ones.

## 🎯 **Core Concept**

The system doesn't just translate obvious column mentions—it **predicts additional relevant columns** that would enhance analysis quality based on:

1. **Query Intent Analysis** - Understanding what type of analysis the user wants
2. **Semantic Similarity** - Finding columns semantically related to the query
3. **Business Logic Rules** - Applying domain-specific knowledge about useful combinations
4. **Metadata Relationships** - Leveraging our rich column metadata for predictions

## 🏗️ **System Architecture**

### **1. Query Analysis Pipeline**

```python
Natural Language Query → Intent Classification → Entity Extraction → Column Prediction → SQL Generation
```

#### **Query Intent Types**
- **`AGGREGATION`** - SUM, COUNT, AVG queries → predict grouping dimensions
- **`COMPARISON`** - Compare groups → predict demographic splits
- **`TREND_ANALYSIS`** - Time-based analysis → predict temporal columns
- **`RANKING`** - TOP/BOTTOM queries → predict ordering columns
- **`CORRELATION`** - Relationship analysis → predict confounding variables
- **`SEGMENTATION`** - Group analysis → predict relevant categories

### **2. Column Prediction Engine**

#### **A. Explicit Column Detection**
```python
"Show customer age and income" → [customer_age_years, income_annual_usd]
```

#### **B. Semantic Similarity Matching**
Using TF-IDF embeddings of column metadata:
```python
Query: "purchasing behavior analysis"
Matches: purchase_history, transaction_amount, customer_lifetime_value
Reasoning: High semantic similarity to metadata descriptions
```

#### **C. Business Logic Rules**
Intent-driven predictions:
```python
AGGREGATION queries → Always include [id, timestamp, grouping_dimensions]
COMPARISON queries → Include [demographics, categorical_splits, performance_metrics]
TREND_ANALYSIS → Require [temporal_columns, seasonality_indicators]
```

#### **D. Metadata Relationship Traversal**
```python
Explicit: customer_age_years
Related: income_annual_usd (via metadata.related_columns)
Correlation: education_level (via metadata.correlation_notes)
```

## 📊 **Example Transformations**

### **Example 1: Simple Aggregation with Enhancement**

**Natural Language:**
```
"Show me the average income by customer segment"
```

**Analysis:**
- Intent: `AGGREGATION`
- Explicit: `income_annual_usd`, `customer_segment`
- Predicted: `customer_age_years` (related), `transaction_date` (business rule)

**Generated SQL:**
```sql
SELECT customer_segment,
       AVG(income_annual_usd) as avg_income_annual_usd,
       SUM(income_annual_usd) as total_income_annual_usd,
       customer_age_years,
       transaction_date
FROM customer_data
GROUP BY customer_segment, customer_age_years, transaction_date
```

### **Example 2: Comparison Query with Demographics**

**Natural Language:**
```
"Compare income between Premium and Standard customers"
```

**Analysis:**
- Intent: `COMPARISON`
- Explicit: `income_annual_usd`, `customer_segment`
- Predicted: `customer_age_years` (demographic), `transaction_date` (temporal context)

**Generated SQL:**
```sql
SELECT customer_segment,
       income_annual_usd,
       customer_age_years,
       transaction_date
FROM customer_data
WHERE customer_segment IN ('Premium', 'Standard')
```

### **Example 3: Trend Analysis with Context**

**Natural Language:**
```
"What's the trend in income over time?"
```

**Analysis:**
- Intent: `TREND_ANALYSIS`  
- Explicit: `income_annual_usd`
- Predicted: `transaction_date` (required temporal), `customer_segment` (context factor)

**Generated SQL:**
```sql
SELECT transaction_date,
       income_annual_usd,
       customer_segment
FROM customer_data
ORDER BY transaction_date
```

## 🔧 **Implementation Components**

### **1. QueryContext Dataclass**
```python
@dataclass
class QueryContext:
    original_query: str
    intent: QueryIntent
    entities: List[str]           # Extracted business entities
    metrics: List[str]            # Numerical measures
    dimensions: List[str]         # Categorical groupings
    temporal_indicators: List[str] # Time-related terms
    comparison_groups: List[str]  # Groups being compared
```

### **2. ColumnRelevance Scoring**
```python
@dataclass
class ColumnRelevance:
    column_name: str
    relevance_score: float        # 0.0 to 1.0
    reasoning: str               # Human-readable explanation
    is_explicit: bool            # Mentioned in query
    is_predicted: bool           # Predicted as useful
    metadata_match: Optional[str] # How it was matched
```

### **3. Business Rules Engine**
```python
business_rules = {
    QueryIntent.AGGREGATION: {
        'always_include': ['id', 'timestamp', 'date'],
        'prefer_numerical': True,
        'include_grouping_dims': True
    },
    QueryIntent.COMPARISON: {
        'include_demographics': True,
        'include_categorical_splits': True,
        'include_performance_metrics': True
    }
}
```

## 🚀 **Usage Examples**

### **Basic Usage**
```python
from natural_language_to_sql import ColumnPredictor, SQLQueryGenerator
from pytorch_frame_metadata import TorchFrameColumnAnnotator

# Initialize with your metadata
annotator = TorchFrameColumnAnnotator()
# ... add your column annotations ...

predictor = ColumnPredictor(annotator)
generator = SQLQueryGenerator(predictor)

# Generate SQL from natural language
result = generator.generate_sql(
    "Show me top customers by revenue last year",
    table_name="sales_data"
)

print("SQL:", result['sql_query'])
print("Predicted columns:", result['predicted_columns'])
print("Confidence:", result['confidence_score'])
```

### **Advanced Configuration**
```python
# Control column prediction limits
result = generator.generate_sql(
    query="Customer analysis by region",
    table_name="customer_data", 
    max_predicted_columns=3  # Limit predictions
)

# Access detailed explanations
for col, explanation in result['column_explanations'].items():
    print(f"{col}: {explanation}")
```

## 🎛️ **Configuration Options**

### **Semantic Similarity Thresholds**
```python
class ColumnPredictor:
    def _find_semantic_matches(self, context):
        # Adjust similarity threshold
        if similarity > 0.1:  # Lower = more permissive
            matches.append((col_name, similarity, reasoning))
```

### **Business Rule Customization**
```python
# Add domain-specific rules
predictor.business_rules[QueryIntent.FINANCIAL_ANALYSIS] = {
    'always_include': ['account_id', 'fiscal_period'],
    'require_monetary_context': True,
    'include_regulatory_fields': True
}
```

### **Column Limit Controls**
```python
# Control prediction volume
result = generator.generate_sql(
    query, 
    max_predicted_columns=5,    # Max additional columns
    confidence_threshold=0.6    # Min confidence for inclusion
)
```

## 🧪 **Testing and Validation**

### **Run Demo**
```python
python natural_language_to_sql.py
```

This demonstrates:
- Intent classification accuracy
- Column prediction reasoning
- SQL generation quality
- Confidence scoring

### **Custom Test Cases**
```python
test_queries = [
    "Revenue analysis by product category",
    "Find anomalous customer behavior patterns", 
    "Compare Q1 vs Q2 performance metrics",
    "Top 10 customers by lifetime value"
]

for query in test_queries:
    result = generator.generate_sql(query)
    # Validate results...
```

## 🎯 **Best Practices**

### **1. Rich Column Metadata**
```python
# Provide comprehensive metadata for better predictions
annotator.annotate_column(
    "customer_lifetime_value_usd",
    description="Total predicted revenue from customer relationship",
    business_meaning="Key metric for customer prioritization and retention",
    related_columns=["customer_segment", "acquisition_cost", "churn_risk"],
    correlation_notes="Strong positive correlation with customer_age and income"
)
```

### **2. Domain-Specific Vocabularies**
```python
# Extend entity extraction for your domain
def _extract_financial_entities(self, query):
    financial_terms = ['revenue', 'profit', 'EBITDA', 'cash_flow']
    # Map to actual column names...
```

### **3. Query Result Validation**
```python
# Always validate generated SQL
def validate_sql_syntax(sql_query):
    try:
        # Parse with sqlparse or similar
        return True
    except:
        return False

# Test against sample data
def test_sql_execution(sql_query, sample_data):
    try:
        result = pd.read_sql(sql_query, connection)
        return len(result) > 0
    except:
        return False
```

### **4. Iterative Improvement**
```python
# Log predictions for analysis
class SQLQueryGenerator:
    def generate_sql(self, query):
        result = super().generate_sql(query)
        
        # Log for improvement
        self._log_prediction(query, result)
        return result
    
    def _log_prediction(self, query, result):
        log_entry = {
            'timestamp': datetime.now(),
            'query': query,
            'intent': result['query_context'].intent,
            'predicted_columns': result['predicted_columns'],
            'confidence': result['confidence_score']
        }
        # Save for analysis...
```

## 🔮 **Advanced Extensions**

### **1. Multi-Table Support**
```python
# Extend for JOIN predictions
def predict_join_tables(self, context, primary_table):
    # Use foreign key relationships in metadata
    # Predict useful JOINs based on query intent
```

### **2. Query Optimization Hints**
```python
# Add performance considerations
def optimize_query_structure(self, sql_query, table_stats):
    # Suggest indexes
    # Reorder JOINs
    # Add LIMIT clauses for large results
```

### **3. Interactive Refinement**
```python
# Allow user feedback on predictions
def refine_prediction(self, query, user_feedback):
    # Learn from user corrections
    # Adjust confidence thresholds
    # Update business rules
```

---

## 🎉 **Key Benefits**

✅ **Enhanced Analysis Quality** - Includes relevant context automatically  
✅ **Reduced Query Iteration** - Gets useful columns on first try  
✅ **Domain Knowledge Integration** - Leverages business relationships  
✅ **Explainable Predictions** - Clear reasoning for each column  
✅ **Extensible Framework** - Easy to customize for specific domains  

This system transforms basic natural language queries into rich, analytically useful SQL by intelligently predicting the columns that would enhance the analysis beyond what the user explicitly requested. 