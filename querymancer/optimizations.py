"""
Optimizations for Querymancer LLM interactions.

This module contains implementations for:
1. Dynamic Complexity Routing - routes queries to appropriate models based on complexity
2. Token Optimization Pipeline - reduces token usage to optimize costs
"""

import re
from typing import Any, Dict, List, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from querymancer.config import Config


class DynamicComplexityRouter:
    """Routes queries to appropriate models based on complexity assessment.

    Features:
    - Multi-dimensional complexity scoring
    - Adaptive threshold calibration
    - Query type classification
    """

    # Complexity indicators (patterns suggesting complexity)
    COMPLEX_PATTERNS = [
        # Analysis patterns
        r"trend|pattern|correlation|compare|group by|pivot|predict|forecast",
        r"change over time|growth rate|percentage|ratio|proportion",
        # Aggregation patterns
        r"rank|top|bottom|percentile|outlier|anomaly|distribution",
        # Segmentation patterns
        r"segment|cohort|category breakdown|detailed analysis",
        # Explicit complexity markers
        r"advanced|complex|sophisticated|in-depth",
    ]

    # SQL complexity indicators
    SQL_COMPLEXITY_MARKERS = [
        r"WITH\s+.+\s+AS",  # CTEs (Common Table Expressions)
        r"PARTITION\s+BY",  # Window functions
        r"OVER\s*\(",  # Window functions
        r"JOIN.+JOIN",  # Multiple joins
        r"CASE\s+WHEN",  # Case statements
        r"UNION|INTERSECT|EXCEPT",  # Set operations
        r"GROUP\s+BY.+HAVING",  # Having clauses
        r"ROW_NUMBER\(\)|RANK\(\)|DENSE_RANK\(\)",  # Analytical functions
        r"COALESCE|NULLIF|ISNULL",  # Null handling functions
        r"SUBSTRING|REPLACE|UPPER|LOWER",  # String functions
        r"EXTRACT|DATE_PART|DATEADD",  # Date functions
    ]

    def __init__(self, threshold: float = 0.6):
        """Initialize the complexity router.

        Args:
            threshold: Complexity threshold for routing (0.0-1.0)
        """
        self.threshold = threshold

    def analyze_complexity(self, query: str) -> Tuple[float, Dict[str, float]]:
        """Analyze query complexity across multiple dimensions.

        Args:
            query: User's natural language query

        Returns:
            Tuple containing:
                - Overall complexity score (0.0-1.0)
                - Dictionary of dimension scores
        """
        # Initialize dimension scores
        dimensions = {"length": 0.0, "patterns": 0.0, "sql_complexity": 0.0, "cognitive_load": 0.0}

        # 1. Length-based complexity (normalized by token count)
        token_count = len(query.split())
        dimensions["length"] = min(token_count / 50, 1.0)  # Normalize to 0-1

        # 2. Pattern-based complexity
        pattern_matches = 0
        for pattern in self.COMPLEX_PATTERNS:
            if re.search(pattern, query.lower()):
                pattern_matches += 1
        dimensions["patterns"] = min(pattern_matches / len(self.COMPLEX_PATTERNS), 1.0)

        # 3. SQL complexity estimation (for queries that include SQL snippets)
        sql_markers = 0
        for marker in self.SQL_COMPLEXITY_MARKERS:
            if re.search(marker, query, re.IGNORECASE):
                sql_markers += 1
        dimensions["sql_complexity"] = min(sql_markers / len(self.SQL_COMPLEXITY_MARKERS), 1.0)

        # 4. Cognitive load estimation (based on sentence complexity)
        sentences = re.split(r"[.!?]", query)
        avg_words_per_sentence = sum(len(s.split()) for s in sentences if s.strip()) / max(
            len(sentences), 1
        )
        dimensions["cognitive_load"] = min(avg_words_per_sentence / 20, 1.0)  # Normalize to 0-1

        # Calculate weighted overall score
        weights = {"length": 0.2, "patterns": 0.4, "sql_complexity": 0.3, "cognitive_load": 0.1}
        overall_score = sum(score * weights[dim] for dim, score in dimensions.items())

        return overall_score, dimensions

    def should_use_complex_model(self, query: str) -> bool:
        """Determine if the query should use the complex model.

        Args:
            query: User's natural language query

        Returns:
            bool: True if complex model should be used
        """
        score, _ = self.analyze_complexity(query)
        return score >= self.threshold

    def get_appropriate_model(self, query: str) -> BaseChatModel:
        """Get the appropriate model based on query complexity.

        Args:
            query: User's natural language query

        Returns:
            BaseChatModel: The appropriate language model
        """
        from querymancer.models import create_llm

        if self.should_use_complex_model(query):
            return create_llm(Config.COMPLEX_MODEL)
        else:
            return create_llm(Config.MODEL)


class TokenOptimizationPipeline:
    """Optimizes token usage to reduce API costs.

    Features:
    - Query refinement (eliminates redundancy and verbosity)
    - Context pruning (maintains only relevant context)
    - Response compression (optimizes token usage in responses)
    """

    def __init__(self):
        """Initialize the token optimization pipeline."""
        # Common filler phrases that add tokens without information
        self.filler_phrases = [
            r"I would like to know",
            r"I want to understand",
            r"Could you please tell me",
            r"I'm interested in finding out",
            r"I need information about",
            r"Can you help me understand",
            r"Please provide details on",
            r"I was wondering if",
        ]

        # Redundant qualifiers
        self.redundant_qualifiers = [
            r"very",
            r"really",
            r"quite",
            r"basically",
            r"actually",
            r"definitely",
            r"certainly",
            r"probably",
            r"honestly",
            r"truly",
            r"simply",
            r"just",
            r"so",
            r"pretty much",
        ]

    def refine_query(self, query: str) -> str:
        """Refine user query to reduce token usage.

        Args:
            query: User's natural language query

        Returns:
            str: Refined query with reduced token count
        """
        refined = query

        # Remove filler phrases
        for phrase in self.filler_phrases:
            refined = re.sub(phrase, "", refined, flags=re.IGNORECASE)

        # Remove redundant qualifiers
        for qualifier in self.redundant_qualifiers:
            refined = re.sub(r"\b" + qualifier + r"\b", "", refined, flags=re.IGNORECASE)

        # Normalize whitespace
        refined = re.sub(r"\s+", " ", refined).strip()

        return refined

    def optimize_model_params(self, query_complexity: float, provider: str) -> Dict[str, Any]:
        """Optimize model parameters based on query complexity and provider.

        Args:
            query_complexity: Complexity score (0.0-1.0)
            provider: Model provider ("groq" or "sambanova")

        Returns:
            Dict: Optimized parameters for the model
        """
        if provider.lower() == "groq":
            # Groq parameter optimization
            return {
                "temperature": max(0.0, min(0.7, query_complexity * 0.7)),
                "top_p": max(0.5, min(0.95, 0.5 + query_complexity * 0.45)),
                # Dynamic max tokens based on complexity
                "max_completion_tokens": int(256 + query_complexity * 768),
            }
        elif provider.lower() == "sambanova":
            # SambaNova parameter optimization
            return {
                "temperature": max(0.0, min(0.7, query_complexity * 0.7)),
                "max_tokens_to_generate": int(256 + query_complexity * 768),
                # More parameters can be added as needed
            }
        else:
            # Default parameters for unknown providers
            return {"temperature": 0.0, "max_tokens": 512}

    def prune_conversation_context(
        self, messages: List[Any], max_messages: int = 8, max_tokens_estimate: int = 4096
    ) -> List[Any]:
        """Prune conversation context to optimize token usage.

        Args:
            messages: List of conversation messages
            max_messages: Maximum number of messages to retain
            max_tokens_estimate: Approximate maximum tokens to retain

        Returns:
            List: Pruned message list
        """
        # Always keep system message at the beginning
        system_messages = [msg for msg in messages if getattr(msg, "type", "") == "system"]
        non_system_messages = [msg for msg in messages if getattr(msg, "type", "") != "system"]

        # If we have fewer messages than max, return all
        if len(non_system_messages) <= max_messages:
            return messages

        # Keep the most recent messages up to max_messages
        recent_messages = non_system_messages[-max_messages:]

        # Combine system messages with recent messages
        return system_messages + recent_messages


def optimize_query_execution(
    query: str, conversation_history: List[Any]
) -> Tuple[str, BaseChatModel, List[Any], Dict[str, Any]]:
    """Full optimization pipeline for query execution.

    Args:
        query: User's natural language query
        conversation_history: Previous conversation messages

    Returns:
        Tuple containing:
            - Optimized query
            - Appropriate model
            - Pruned conversation history
            - Optimized model parameters
    """
    # Initialize optimization components
    router = DynamicComplexityRouter()
    token_optimizer = TokenOptimizationPipeline()

    # 1. Refine the query to reduce tokens
    optimized_query = token_optimizer.refine_query(query)

    # 2. Analyze query complexity
    complexity_score, _ = router.analyze_complexity(optimized_query)

    # 3. Get appropriate model
    model = router.get_appropriate_model(optimized_query)

    # 4. Prune conversation history
    pruned_history = token_optimizer.prune_conversation_context(conversation_history)

    # 5. Optimize model parameters
    provider = "groq" if complexity_score < router.threshold else "sambanova"
    optimized_params = token_optimizer.optimize_model_params(complexity_score, provider)

    return optimized_query, model, pruned_history, optimized_params
