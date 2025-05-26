"""
Natural Language to SQL Translation with Intelligent Column Prediction
Leverages column metadata framework to predict useful columns beyond obvious ones
"""

import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from enum import Enum
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class QueryIntent(Enum):
    """Classification of query intents for column prediction"""
    AGGREGATION = "aggregation"  # SUM, COUNT, AVG queries
    FILTERING = "filtering"      # WHERE conditions
    COMPARISON = "comparison"    # Compare groups/segments
    TREND_ANALYSIS = "trend"     # Time-based analysis
    CORRELATION = "correlation"  # Relationship analysis
    SEGMENTATION = "segmentation" # Group analysis
    ANOMALY = "anomaly"          # Outlier detection
    RANKING = "ranking"          # TOP/BOTTOM queries

@dataclass
class ColumnRelevance:
    """Score and reasoning for column relevance"""
    column_name: str
    relevance_score: float
    reasoning: str
    is_explicit: bool = False  # Mentioned in query
    is_predicted: bool = False # Predicted as useful
    metadata_match: Optional[str] = None

@dataclass
class QueryContext:
    """Rich context about the user's query"""
    original_query: str
    intent: QueryIntent
    entities: List[str]
    metrics: List[str]
    dimensions: List[str]
    temporal_indicators: List[str]
    comparison_groups: List[str]

class ColumnPredictor:
    """Predicts useful columns based on query analysis and metadata"""
    
    def __init__(self, metadata_annotator):
        self.metadata = metadata_annotator
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.business_rules = self._initialize_business_rules()
        self.column_embeddings = None
        self._prepare_column_embeddings()
    
    def _prepare_column_embeddings(self):
        """Create semantic embeddings for all columns"""
        if not self.metadata.column_info:
            return
            
        # Combine column metadata for semantic matching
        column_texts = []
        column_names = []
        
        for col_name, meta in self.metadata.column_info.items():
            # Create rich text representation of column
            text_parts = [
                col_name,
                meta.get('description', ''),
                meta.get('business_meaning', ''),
                meta.get('torch_frame_stype', '')
            ]
            
            # Add categories if they exist
            categories = meta.get('categories', [])
            if categories:
                text_parts.append(' '.join(str(cat) for cat in categories))
            
            column_text = ' '.join(filter(None, text_parts))
            column_texts.append(column_text)
            column_names.append(col_name)
        
        if column_texts:
            self.column_embeddings = self.vectorizer.fit_transform(column_texts)
            self.column_names = column_names
    
    def _initialize_business_rules(self):
        """Define business logic rules for column prediction"""
        return {
            QueryIntent.AGGREGATION: {
                'always_include': ['id', 'timestamp', 'date'],
                'prefer_numerical': True,
                'include_grouping_dims': True
            },
            QueryIntent.COMPARISON: {
                'include_demographics': True,
                'include_categorical_splits': True,
                'include_performance_metrics': True
            },
            QueryIntent.TREND_ANALYSIS: {
                'require_temporal': True,
                'include_seasonality_indicators': True,
                'include_context_factors': True
            },
            QueryIntent.CORRELATION: {
                'include_related_metrics': True,
                'include_potential_confounders': True,
                'avoid_derived_from_same_source': True
            }
        }
    
    def analyze_query_intent(self, query: str) -> QueryContext:
        """Analyze natural language query to understand intent and entities"""
        query_lower = query.lower()
        
        # Intent classification
        intent = self._classify_intent(query_lower)
        
        # Entity extraction
        entities = self._extract_entities(query)
        metrics = self._extract_metrics(query)
        dimensions = self._extract_dimensions(query)
        temporal_indicators = self._extract_temporal_indicators(query)
        comparison_groups = self._extract_comparison_groups(query)
        
        return QueryContext(
            original_query=query,
            intent=intent,
            entities=entities,
            metrics=metrics,
            dimensions=dimensions,
            temporal_indicators=temporal_indicators,
            comparison_groups=comparison_groups
        )
    
    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent using keyword patterns"""
        intent_patterns = {
            QueryIntent.AGGREGATION: [
                'total', 'sum', 'count', 'average', 'mean', 'aggregate',
                'how many', 'how much', 'total number'
            ],
            QueryIntent.COMPARISON: [
                'compare', 'versus', 'vs', 'difference between', 'higher than',
                'lower than', 'better than', 'worse than'
            ],
            QueryIntent.TREND_ANALYSIS: [
                'trend', 'over time', 'monthly', 'yearly', 'growth', 'change',
                'increase', 'decrease', 'pattern'
            ],
            QueryIntent.RANKING: [
                'top', 'bottom', 'highest', 'lowest', 'best', 'worst',
                'rank', 'ranking', 'order by'
            ],
            QueryIntent.SEGMENTATION: [
                'group by', 'segment', 'category', 'type', 'breakdown'
            ]
        }
        
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query)
            if score > 0:
                intent_scores[intent] = score
        
        return max(intent_scores, key=intent_scores.get) if intent_scores else QueryIntent.FILTERING
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract business entities mentioned in query"""
        # Match against column names and metadata
        entities = []
        query_lower = query.lower()
        
        for col_name, meta in self.metadata.column_info.items():
            # Check column name
            if col_name.lower() in query_lower:
                entities.append(col_name)
            
            # Check description and business meaning
            description = meta.get('description', '').lower()
            business_meaning = meta.get('business_meaning', '').lower()
            
            if description and any(word in query_lower for word in description.split()):
                entities.append(col_name)
        
        return list(set(entities))
    
    def _extract_metrics(self, query: str) -> List[str]:
        """Extract metrics/measurements from query"""
        metric_keywords = [
            'revenue', 'sales', 'profit', 'cost', 'price', 'amount',
            'count', 'quantity', 'rate', 'percentage', 'score',
            'performance', 'efficiency', 'productivity'
        ]
        
        metrics = []
        query_lower = query.lower()
        
        for keyword in metric_keywords:
            if keyword in query_lower:
                # Find columns related to this metric
                for col_name, meta in self.metadata.column_info.items():
                    if (keyword in meta.get('description', '').lower() or
                        keyword in meta.get('business_meaning', '').lower()):
                        metrics.append(col_name)
        
        return list(set(metrics))
    
    def _extract_dimensions(self, query: str) -> List[str]:
        """Extract dimensional attributes for grouping"""
        dimensions = []
        
        for col_name, meta in self.metadata.column_info.items():
            if meta.get('torch_frame_stype') == 'categorical':
                # Check if mentioned in query
                if (col_name.lower() in query.lower() or
                    any(cat.lower() in query.lower() 
                        for cat in meta.get('categories', []))):
                    dimensions.append(col_name)
        
        return dimensions
    
    def _extract_temporal_indicators(self, query: str) -> List[str]:
        """Extract time-related terms"""
        temporal_patterns = [
            'today', 'yesterday', 'last week', 'this month', 'this year',
            'daily', 'weekly', 'monthly', 'yearly', 'over time',
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'q1', 'q2', 'q3', 'q4', 'quarter'
        ]
        
        indicators = []
        query_lower = query.lower()
        
        for pattern in temporal_patterns:
            if pattern in query_lower:
                indicators.append(pattern)
        
        return indicators
    
    def _extract_comparison_groups(self, query: str) -> List[str]:
        """Extract groups being compared"""
        comparison_patterns = [
            r'(\w+)\s+vs\s+(\w+)',
            r'(\w+)\s+versus\s+(\w+)',
            r'compare\s+(\w+)\s+and\s+(\w+)',
            r'(\w+)\s+compared\s+to\s+(\w+)'
        ]
        
        groups = []
        for pattern in comparison_patterns:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                groups.extend(match)
        
        return groups
    
    def predict_useful_columns(self, context: QueryContext) -> List[ColumnRelevance]:
        """Main method to predict useful columns beyond obvious ones"""
        column_scores = {}
        
        # 1. Score explicitly mentioned columns
        explicit_columns = set()
        for entity in context.entities + context.metrics + context.dimensions:
            if entity in self.metadata.column_info:
                explicit_columns.add(entity)
                column_scores[entity] = ColumnRelevance(
                    column_name=entity,
                    relevance_score=1.0,
                    reasoning="Explicitly mentioned in query",
                    is_explicit=True
                )
        
        # 2. Apply semantic similarity matching
        semantic_matches = self._find_semantic_matches(context)
        for col_name, score, reasoning in semantic_matches:
            if col_name not in explicit_columns:
                column_scores[col_name] = ColumnRelevance(
                    column_name=col_name,
                    relevance_score=score,
                    reasoning=reasoning,
                    is_predicted=True,
                    metadata_match="semantic_similarity"
                )
        
        # 3. Apply business rules
        business_matches = self._apply_business_rules(context, explicit_columns)
        for col_name, score, reasoning in business_matches:
            if col_name not in column_scores:
                column_scores[col_name] = ColumnRelevance(
                    column_name=col_name,
                    relevance_score=score,
                    reasoning=reasoning,
                    is_predicted=True,
                    metadata_match="business_rules"
                )
        
        # 4. Add related columns through metadata relationships
        related_matches = self._find_related_columns(explicit_columns)
        for col_name, score, reasoning in related_matches:
            if col_name not in column_scores:
                column_scores[col_name] = ColumnRelevance(
                    column_name=col_name,
                    relevance_score=score,
                    reasoning=reasoning,
                    is_predicted=True,
                    metadata_match="metadata_relationships"
                )
        
        # 5. Sort by relevance score
        return sorted(column_scores.values(), 
                     key=lambda x: x.relevance_score, reverse=True)
    
    def _find_semantic_matches(self, context: QueryContext) -> List[Tuple[str, float, str]]:
        """Find columns using semantic similarity"""
        if self.column_embeddings is None:
            return []
        
        # Create query embedding
        query_embedding = self.vectorizer.transform([context.original_query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.column_embeddings)[0]
        
        matches = []
        for i, similarity in enumerate(similarities):
            if similarity > 0.1:  # Threshold for relevance
                col_name = self.column_names[i]
                reasoning = f"Semantic similarity score: {similarity:.3f}"
                matches.append((col_name, similarity, reasoning))
        
        return matches
    
    def _apply_business_rules(self, context: QueryContext, 
                            explicit_columns: Set[str]) -> List[Tuple[str, float, str]]:
        """Apply business logic rules for column prediction"""
        rules = self.business_rules.get(context.intent, {})
        matches = []
        
        # Always include certain column types
        always_include = rules.get('always_include', [])
        for col_pattern in always_include:
            for col_name, meta in self.metadata.column_info.items():
                if (col_pattern in col_name.lower() and 
                    col_name not in explicit_columns):
                    matches.append((col_name, 0.8, 
                                  f"Business rule: always include {col_pattern}"))
        
        # Intent-specific rules
        if context.intent == QueryIntent.AGGREGATION:
            # Include numerical columns for aggregation
            for col_name, meta in self.metadata.column_info.items():
                if (meta.get('torch_frame_stype') == 'numerical' and 
                    col_name not in explicit_columns):
                    matches.append((col_name, 0.6, 
                                  "Numerical column useful for aggregation"))
        
        elif context.intent == QueryIntent.COMPARISON:
            # Include demographic and categorical columns
            for col_name, meta in self.metadata.column_info.items():
                if (meta.get('torch_frame_stype') == 'categorical' and 
                    'demographic' in meta.get('business_meaning', '').lower()):
                    matches.append((col_name, 0.7, 
                                  "Demographic dimension for comparison"))
        
        elif context.intent == QueryIntent.TREND_ANALYSIS:
            # Include time-related columns
            for col_name, meta in self.metadata.column_info.items():
                if ('date' in col_name.lower() or 'time' in col_name.lower() or
                    meta.get('torch_frame_stype') == 'timestamp'):
                    matches.append((col_name, 0.9, 
                                  "Temporal column required for trend analysis"))
        
        return matches
    
    def _find_related_columns(self, explicit_columns: Set[str]) -> List[Tuple[str, float, str]]:
        """Find columns related to explicitly mentioned ones"""
        matches = []
        
        # Note: Basic metadata doesn't include relationships, but we can infer some
        for explicit_col in explicit_columns:
            if explicit_col in self.metadata.column_info:
                meta = self.metadata.column_info[explicit_col]
                
                # Find columns with similar business meaning
                business_meaning = meta.get('business_meaning', '').lower()
                if business_meaning:
                    for col_name, other_meta in self.metadata.column_info.items():
                        if col_name != explicit_col:
                            other_meaning = other_meta.get('business_meaning', '').lower()
                            # Simple keyword overlap check
                            if any(word in other_meaning for word in business_meaning.split() 
                                   if len(word) > 4):  # Only meaningful words
                                matches.append((col_name, 0.5, 
                                              f"Similar business context to {explicit_col}"))
        
        return matches

class SQLQueryGenerator:
    """Generates SQL queries with predicted columns"""
    
    def __init__(self, column_predictor: ColumnPredictor):
        self.predictor = column_predictor
        
    def generate_sql(self, natural_query: str, 
                    table_name: str = "dataset",
                    max_predicted_columns: int = 5) -> Dict[str, Any]:
        """Generate SQL query with explanation"""
        
        # Analyze the query
        context = self.predictor.analyze_query_intent(natural_query)
        
        # Predict useful columns
        column_relevance = self.predictor.predict_useful_columns(context)
        
        # Separate explicit and predicted columns
        explicit_cols = [col for col in column_relevance if col.is_explicit]
        predicted_cols = [col for col in column_relevance 
                         if col.is_predicted][:max_predicted_columns]
        
        # Generate SQL components
        select_clause = self._generate_select_clause(
            context, explicit_cols + predicted_cols)
        where_clause = self._generate_where_clause(context)
        group_by_clause = self._generate_group_by_clause(context)
        order_by_clause = self._generate_order_by_clause(context)
        
        # Construct full SQL
        sql_parts = [f"SELECT {select_clause}"]
        sql_parts.append(f"FROM {table_name}")
        
        if where_clause:
            sql_parts.append(f"WHERE {where_clause}")
        
        if group_by_clause:
            sql_parts.append(f"GROUP BY {group_by_clause}")
        
        if order_by_clause:
            sql_parts.append(f"ORDER BY {order_by_clause}")
        
        sql_query = "\n".join(sql_parts)
        
        return {
            'sql_query': sql_query,
            'query_context': context,
            'explicit_columns': [col.column_name for col in explicit_cols],
            'predicted_columns': [col.column_name for col in predicted_cols],
            'column_explanations': {
                col.column_name: col.reasoning 
                for col in explicit_cols + predicted_cols
            },
            'confidence_score': self._calculate_confidence(column_relevance)
        }
    
    def _generate_select_clause(self, context: QueryContext, 
                               columns: List[ColumnRelevance]) -> str:
        """Generate SELECT clause based on intent and columns"""
        
        if context.intent == QueryIntent.AGGREGATION:
            # Include aggregation functions
            select_parts = []
            
            # Add grouping dimensions
            for col in columns:
                meta = self.predictor.metadata.column_info.get(col.column_name, {})
                if meta.get('torch_frame_stype') == 'categorical':
                    select_parts.append(col.column_name)
            
            # Add aggregated metrics
            for col in columns:
                meta = self.predictor.metadata.column_info.get(col.column_name, {})
                if meta.get('torch_frame_stype') == 'numerical':
                    select_parts.append(f"AVG({col.column_name}) as avg_{col.column_name}")
                    select_parts.append(f"SUM({col.column_name}) as total_{col.column_name}")
            
            return ", ".join(select_parts) if select_parts else "*"
        
        elif context.intent == QueryIntent.RANKING:
            # Include ranking columns
            column_names = [col.column_name for col in columns]
            return ", ".join(column_names)
        
        else:
            # Default: select all relevant columns
            column_names = [col.column_name for col in columns]
            return ", ".join(column_names) if column_names else "*"
    
    def _generate_where_clause(self, context: QueryContext) -> str:
        """Generate WHERE clause based on query context"""
        conditions = []
        
        # Extract filter conditions from natural language
        query_lower = context.original_query.lower()
        
        # Simple pattern matching for common filters
        if 'last year' in query_lower:
            conditions.append("YEAR(date_column) = YEAR(CURRENT_DATE) - 1")
        elif 'this year' in query_lower:
            conditions.append("YEAR(date_column) = YEAR(CURRENT_DATE)")
        
        # Handle comparison groups
        for group in context.comparison_groups:
            # This would need more sophisticated NLP
            pass
        
        return " AND ".join(conditions) if conditions else ""
    
    def _generate_group_by_clause(self, context: QueryContext) -> str:
        """Generate GROUP BY clause"""
        if context.intent in [QueryIntent.AGGREGATION, QueryIntent.COMPARISON]:
            return ", ".join(context.dimensions) if context.dimensions else ""
        return ""
    
    def _generate_order_by_clause(self, context: QueryContext) -> str:
        """Generate ORDER BY clause"""
        if context.intent == QueryIntent.RANKING:
            # Infer ordering from query
            query_lower = context.original_query.lower()
            if 'highest' in query_lower or 'top' in query_lower:
                return f"{context.metrics[0]} DESC" if context.metrics else ""
            elif 'lowest' in query_lower or 'bottom' in query_lower:
                return f"{context.metrics[0]} ASC" if context.metrics else ""
        return ""
    
    def _calculate_confidence(self, column_relevance: List[ColumnRelevance]) -> float:
        """Calculate confidence score for the generated query"""
        if not column_relevance:
            return 0.0
        
        explicit_weight = 0.8
        predicted_weight = 0.2
        
        total_score = 0
        total_weight = 0
        
        for col in column_relevance:
            weight = explicit_weight if col.is_explicit else predicted_weight
            total_score += col.relevance_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

# Usage Example and Demo
def demo_natural_language_to_sql():
    """Demonstrate the natural language to SQL system"""
    
    # Create sample metadata (in practice, load from your metadata system)
    from pytorch_frame_metadata import TorchFrameColumnAnnotator
    
    annotator = TorchFrameColumnAnnotator()
    
    # Add sample column metadata (using actual API)
    annotator.annotate_column(
        "customer_age_years",
        description="Age of customer in years",
        torch_frame_stype="numerical",
        purpose="feature",
        business_meaning="Age strongly correlates with income and purchasing behavior"
    )
    
    annotator.annotate_column(
        "income_annual_usd", 
        description="Annual gross income in USD",
        torch_frame_stype="numerical",
        purpose="feature",
        business_meaning="Primary indicator of purchasing power and credit worthiness"
    )
    
    annotator.annotate_column(
        "customer_segment",
        description="Customer segmentation category",
        torch_frame_stype="categorical", 
        purpose="feature",
        categories=["Premium", "Standard", "Basic"],
        business_meaning="Marketing segment based on value and behavior"
    )
    
    annotator.annotate_column(
        "transaction_date",
        description="Date of transaction",
        torch_frame_stype="timestamp",
        purpose="feature", 
        business_meaning="Essential for trend analysis and seasonality detection"
    )
    
    # Initialize the system
    predictor = ColumnPredictor(annotator)
    generator = SQLQueryGenerator(predictor)
    
    # Test queries
    test_queries = [
        "Show me the average income by customer segment",
        "Which customers had the highest income last year?", 
        "Compare income between Premium and Standard customers",
        "What's the trend in income over time?",
        "Find customers with unusual income patterns"
    ]
    
    print("🚀 NATURAL LANGUAGE TO SQL DEMO")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Natural Language Query:")
        print(f"   '{query}'")
        
        result = generator.generate_sql(query, table_name="customer_data")
        
        print(f"\n   Generated SQL:")
        print("   " + "\n   ".join(result['sql_query'].split('\n')))
        
        print(f"\n   Query Intent: {result['query_context'].intent.value}")
        print(f"   Confidence Score: {result['confidence_score']:.3f}")
        
        print(f"\n   Explicit Columns: {result['explicit_columns']}")
        print(f"   Predicted Columns: {result['predicted_columns']}")
        
        print(f"\n   Column Explanations:")
        for col, explanation in result['column_explanations'].items():
            print(f"     {col}: {explanation}")
        
        print("\n" + "-" * 60)

if __name__ == "__main__":
    demo_natural_language_to_sql() 