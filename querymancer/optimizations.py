"""
Optimizations for Querymancer LLM interactions.

This module contains implementations for:
1. Dynamic Complexity Routing - routes queries to appropriate models based on complexity
2. Token Optimization Pipeline - reduces token usage to optimize costs
3. Performance Monitoring - tracks model performance and user feedback
4. Threshold Calibration - dynamically adjusts complexity thresholds based on performance data
"""

import re
from typing import Any, Dict, List, Tuple, Optional
import hashlib
from datetime import datetime, timedelta
import threading
from collections import OrderedDict, defaultdict
import json
import os
import statistics
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from querymancer.config import Config


class ComplexityCache:
    """Cache for query complexity analysis results.

    Implements a simple LRU (Least Recently Used) cache to avoid
    re-analyzing similar queries multiple times.
    """

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """Initialize the complexity cache.

        Args:
            max_size: Maximum number of items to store in the cache
            ttl: Time-to-live in seconds for cache entries
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.RLock()

    def _generate_key(self, query: str) -> str:
        """Generate a cache key for the query.

        Args:
            query: User's natural language query

        Returns:
            str: Cache key
        """
        # Normalize query for better cache hits
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[Tuple[float, Dict[str, float]]]:
        """Get cached complexity result for a query.

        Args:
            query: User's natural language query

        Returns:
            Optional[Tuple[float, Dict[str, float]]]: Cached result or None
        """
        key = self._generate_key(query)

        with self.lock:
            if key not in self.cache:
                return None

            timestamp, result = self.cache[key]
            current_time = datetime.now()

            # Check if the entry is expired
            if current_time - timestamp > timedelta(seconds=self.ttl):
                del self.cache[key]
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return result

    def set(self, query: str, result: Tuple[float, Dict[str, float]]) -> None:
        """Cache complexity result for a query.

        Args:
            query: User's natural language query
            result: Complexity analysis result
        """
        key = self._generate_key(query)

        with self.lock:
            # Add or update the cache
            self.cache[key] = (datetime.now(), result)
            self.cache.move_to_end(key)

            # Trim cache if needed
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)


class PerformanceMonitor:
    """Tracks model performance, selection decisions, and response quality.

    Features:
    - Records model selection decisions
    - Tracks response quality metrics
    - Provides historical performance analysis
    - Supports threshold calibration
    """

    def __init__(self, log_file: str = "model_performance.json"):
        """Initialize the performance monitor.

        Args:
            log_file: Path to the log file for recording performance metrics
        """
        self.log_file = log_file
        self.metrics = defaultdict(list)
        self.lock = threading.RLock()

        # Load existing data if available
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    data = json.load(f)
                    for key, values in data.items():
                        self.metrics[key] = values
            except (json.JSONDecodeError, FileNotFoundError):
                # Start with empty metrics if file doesn't exist or is corrupt
                pass

    def record_selection(self, query: str, complexity: float, selected_model: str) -> None:
        """Record a model selection decision.

        Args:
            query: User query
            complexity: Calculated complexity score
            selected_model: The model that was selected ("groq" or "sambanova")
        """
        with self.lock:
            record = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "complexity": complexity,
                "selected_model": selected_model,
            }
            self.metrics["selections"].append(record)
            self._save_metrics()

    def record_feedback(self, query: str, rating: int, comments: Optional[str] = None) -> None:
        """Record user feedback on response quality.

        Args:
            query: The original user query
            rating: User rating (1-5)
            comments: Optional user comments
        """
        with self.lock:
            record = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "rating": max(1, min(5, rating)),  # Ensure rating is between 1-5
                "comments": comments,
            }
            self.metrics["feedback"].append(record)
            self._save_metrics()

    def get_model_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics by model.

        Returns:
            Dict containing performance metrics for each model
        """
        with self.lock:
            model_metrics = defaultdict(lambda: {"count": 0, "avg_rating": 0.0, "ratings": []})

            # Process feedback with corresponding selections
            selection_map = {}
            for selection in self.metrics.get("selections", []):
                selection_map[selection.get("query", "")] = selection

            for feedback in self.metrics.get("feedback", []):
                query = feedback.get("query", "")
                if query in selection_map:
                    model = selection_map[query].get("selected_model", "unknown")
                    rating = feedback.get("rating", 0)

                    model_metrics[model]["count"] += 1
                    model_metrics[model]["ratings"].append(rating)

            # Calculate averages
            for model, data in model_metrics.items():
                if data["ratings"]:
                    data["avg_rating"] = sum(data["ratings"]) / len(data["ratings"])

                # Add additional statistics if we have enough data
                if len(data["ratings"]) >= 5:
                    data["median_rating"] = statistics.median(data["ratings"])
                    try:
                        data["std_dev"] = statistics.stdev(data["ratings"])
                    except statistics.StatisticsError:
                        data["std_dev"] = 0.0

            return dict(model_metrics)

    def get_complexity_distribution(self) -> Dict[str, List[float]]:
        """Get the distribution of complexity scores by selected model.

        Returns:
            Dict mapping model names to lists of complexity scores
        """
        with self.lock:
            distribution = defaultdict(list)

            for selection in self.metrics.get("selections", []):
                model = selection.get("selected_model", "")
                complexity = selection.get("complexity", 0.0)
                distribution[model].append(complexity)

            return dict(distribution)

    def _save_metrics(self) -> None:
        """Save metrics to the log file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)

            with open(self.log_file, "w") as f:
                json.dump(dict(self.metrics), f, indent=2)
        except (OSError, IOError) as e:
            print(f"Error saving metrics: {e}")


class ThresholdCalibrator:
    """Dynamically calibrates complexity thresholds based on performance data.

    Features:
    - Analyzes historical performance data
    - Suggests optimal threshold adjustments
    - Supports automated or manual calibration
    """

    def __init__(self, initial_threshold: float = 0.25, min_samples: int = 20):
        """Initialize the threshold calibrator.

        Args:
            initial_threshold: Starting threshold value
            min_samples: Minimum samples required for calibration
        """
        self.current_threshold = initial_threshold
        self.min_samples = min_samples
        self.calibration_history = []

    def suggest_threshold(self, performance_monitor: PerformanceMonitor) -> float:
        """Suggest an optimal threshold based on performance data.

        Args:
            performance_monitor: PerformanceMonitor instance with historical data

        Returns:
            float: Suggested complexity threshold
        """
        # Get performance metrics
        metrics = performance_monitor.get_model_performance_metrics()
        complexity_dist = performance_monitor.get_complexity_distribution()

        # Check if we have enough data for meaningful calibration
        groq_ratings = metrics.get("groq", {}).get("ratings", [])
        sambanova_ratings = metrics.get("sambanova", {}).get("ratings", [])

        if len(groq_ratings) < self.min_samples or len(sambanova_ratings) < self.min_samples:
            # Not enough data yet
            return self.current_threshold

        # Calculate average ratings
        groq_avg = sum(groq_ratings) / len(groq_ratings)
        sambanova_avg = sum(sambanova_ratings) / len(sambanova_ratings)

        # Get complexity distributions
        groq_complexities = complexity_dist.get("groq", [])
        sambanova_complexities = complexity_dist.get("sambanova", [])

        if not groq_complexities or not sambanova_complexities:
            return self.current_threshold

        # Find the optimal threshold by analyzing where SambaNova outperforms Groq
        # This is a simplified approach - a real implementation might use more sophisticated methods
        if sambanova_avg > groq_avg:
            # SambaNova is performing better, we might want to route more queries to it
            # by lowering the threshold

            # Find the 75th percentile of Groq complexities
            groq_complexities.sort()
            idx = min(int(len(groq_complexities) * 0.75), len(groq_complexities) - 1)
            suggested = groq_complexities[idx]

            # Ensure we don't make drastic changes
            max_change = 0.05  # Maximum 5% change at once
            delta = suggested - self.current_threshold
            if abs(delta) > max_change:
                delta = max_change if delta > 0 else -max_change

            new_threshold = self.current_threshold + delta

            # Keep threshold in reasonable bounds
            new_threshold = max(0.1, min(0.5, new_threshold))

        else:
            # Groq is performing better, we might want to route more queries to it
            # by raising the threshold

            # Find the 25th percentile of SambaNova complexities
            sambanova_complexities.sort()
            idx = min(int(len(sambanova_complexities) * 0.25), len(sambanova_complexities) - 1)
            suggested = sambanova_complexities[idx]

            # Ensure we don't make drastic changes
            max_change = 0.05  # Maximum 5% change at once
            delta = suggested - self.current_threshold
            if abs(delta) > max_change:
                delta = max_change if delta > 0 else -max_change

            new_threshold = self.current_threshold + delta

            # Keep threshold in reasonable bounds
            new_threshold = max(0.1, min(0.5, new_threshold))

        # Record this calibration event
        self.calibration_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "old_threshold": self.current_threshold,
                "new_threshold": new_threshold,
                "groq_avg_rating": groq_avg,
                "sambanova_avg_rating": sambanova_avg,
                "sample_sizes": {"groq": len(groq_ratings), "sambanova": len(sambanova_ratings)},
            }
        )

        self.current_threshold = new_threshold
        return new_threshold

    def apply_calibration(self, performance_monitor: PerformanceMonitor) -> float:
        """Apply threshold calibration based on performance data.

        Args:
            performance_monitor: PerformanceMonitor instance with historical data

        Returns:
            float: New threshold value
        """
        new_threshold = self.suggest_threshold(performance_monitor)
        self.current_threshold = new_threshold
        return new_threshold

    def get_calibration_history(self) -> List[Dict[str, Any]]:
        """Get the history of threshold calibrations.

        Returns:
            List of calibration events
        """
        return self.calibration_history.copy()


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
        # Additional analytical patterns
        r"analyze|insight|metrics|dashboard|visualization|data mining",
        r"statistics|statistical|regression|classification|clustering",
        r"time series|seasonality|benchmark|performance indicator",
        r"multi-dimensional|cross-tabulation|stratification|aggregation",
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
        # Additional SQL complexity markers
        r"LAG\(\)|LEAD\(\)|FIRST_VALUE\(\)|LAST_VALUE\(\)",  # More window functions
        r"CUBE|ROLLUP",  # Advanced grouping
        r"PIVOT|UNPIVOT",  # Pivoting operations
        r"RECURSIVE",  # Recursive CTEs
        r"JSON_.*\(\)|XML_.*\(\)",  # JSON/XML operations
        r"PERCENTILE_.*\(\)",  # Statistical functions
    ]

    def __init__(self, threshold: float = 0.25):
        """Initialize the complexity router.

        Args:
            threshold: Complexity threshold for routing (0.0-1.0)
        """
        self.threshold = threshold
        self.cache = ComplexityCache()

        # Add performance monitor
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        self.monitor = PerformanceMonitor(log_file=str(logs_dir / "model_performance.json"))

        # Add threshold calibrator
        self.calibrator = ThresholdCalibrator(initial_threshold=threshold)

        # Periodically calibrate threshold (less frequently than recording)
        self._last_calibration = datetime.now()
        self._calibration_interval = timedelta(days=1)  # Calibrate daily

        # Domain-specific complexity modifiers
        self.domain_modifiers = {
            "sql": 0.15,  # SQL queries get a boost in complexity
            "data_science": 0.20,  # Data science queries get a larger boost
            "reporting": 0.10,  # Reporting queries get a moderate boost
            "simple_lookup": -0.05,  # Simple lookups get a reduction in complexity
        }

        # Domain detection patterns
        self.domain_patterns = {
            "sql": [
                r"SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP",
                r"FROM\s+\w+(\s+JOIN)?",
                r"WHERE\s+\w+",
                r"GROUP BY|ORDER BY|HAVING",
            ],
            "data_science": [
                r"machine learning|neural network|deep learning|AI|artificial intelligence",
                r"regression|classification|clustering|NLP|natural language processing",
                r"train|model|algorithm|feature|dataset|prediction|accuracy",
                r"correlation|causation|hypothesis|confidence interval|p-value",
            ],
            "reporting": [
                r"report|dashboard|visualization|chart|graph|plot",
                r"KPI|key performance indicator|metric|measure|benchmark",
                r"daily|weekly|monthly|quarterly|yearly|annual",
                r"trend|comparison|breakdown|summary",
            ],
            "simple_lookup": [
                r"find|get|retrieve|show|list|display",
                r"lookup|search|select",
                r"simple|basic|quick",
                r"single|one|individual",
            ],
        }

    def _detect_domain(self, query: str) -> str:
        """Detect the domain of the query.

        Args:
            query: User's natural language query

        Returns:
            str: Detected domain or None
        """
        domain_scores = {}

        # Calculate score for each domain
        for domain, patterns in self.domain_patterns.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    matches += 1

            domain_scores[domain] = matches / len(patterns) if len(patterns) > 0 else 0

        # Find the domain with the highest score above threshold
        best_domain = None
        best_score = 0.3  # Minimum threshold to assign a domain

        for domain, score in domain_scores.items():
            if score > best_score:
                best_domain = domain
                best_score = score

        return best_domain

    def analyze_complexity(self, query: str) -> Tuple[float, Dict[str, float]]:
        """Analyze query complexity across multiple dimensions.

        Args:
            query: User's natural language query

        Returns:
            Tuple containing:
                - Overall complexity score (0.0-1.0)
                - Dictionary of dimension scores
        """
        # Check cache first
        cached_result = self.cache.get(query)
        if cached_result:
            return cached_result

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
        weights = {"length": 0.15, "patterns": 0.45, "sql_complexity": 0.3, "cognitive_load": 0.1}
        overall_score = sum(score * weights[dim] for dim, score in dimensions.items())

        # Apply domain-specific modifiers
        domain = self._detect_domain(query)
        if domain and domain in self.domain_modifiers:
            modifier = self.domain_modifiers[domain]
            overall_score = min(max(overall_score + modifier, 0.0), 1.0)  # Keep in range [0, 1]

        # Cache the result
        result = (overall_score, dimensions)
        self.cache.set(query, result)

        return result

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

        # Analyze complexity
        complexity_score, _ = self.analyze_complexity(query)

        # Check if threshold calibration is due
        now = datetime.now()
        if now - self._last_calibration >= self._calibration_interval:
            self.threshold = self.calibrator.suggest_threshold(self.monitor)
            self._last_calibration = now

        # Determine which model to use
        use_complex_model = complexity_score >= self.threshold
        selected_model = "sambanova" if use_complex_model else "groq"

        # Record the selection decision
        self.monitor.record_selection(query, complexity_score, selected_model)

        # Create and return the appropriate model
        if use_complex_model:
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
            r"I need information about",
            r"Can you help me understand",
            r"Please provide details on",
            r"I was wondering if",
            # Additional filler phrases
            r"If you don't mind",
            r"I'd appreciate it if",
            r"It would be great if you could",
            r"I'm curious about",
            r"Would it be possible to",
            r"Do you think you could",
            r"I'm trying to figure out",
            r"I would be grateful if",
            r"If it's not too much to ask",
            r"I'd like to know more about",
            r"Could you explain to me",
            r"Hi, I need",
            r"Hello, can you",
            r"Please help me with",
            r"If you have time",
            r"I'm looking for",
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
            # Additional redundant qualifiers
            r"super",
            r"extremely",
            r"incredibly",
            r"absolutely",
            r"totally",
            r"entirely",
            r"completely",
            r"utterly",
            r"rather",
            r"somewhat",
            r"kind of",
            r"sort of",
            r"a bit",
            r"a little",
            r"in my opinion",
            r"I think",
            r"I believe",
            r"as far as I know",
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

        # Detect domain for domain-specific optimizations
        router = DynamicComplexityRouter()
        domain = router._detect_domain(refined)

        # Apply domain-specific optimizations if applicable
        if domain:
            refined = self.optimize_specific_domain(refined, domain)

        # Normalize whitespace
        refined = re.sub(r"\s+", " ", refined).strip()

        return refined

    def optimize_specific_domain(self, query: str, domain: str) -> str:
        """Apply domain-specific optimizations to the query.

        Args:
            query: User's natural language query
            domain: Domain identifier (e.g., "sql", "data_science")

        Returns:
            str: Optimized query
        """
        if domain == "sql":
            # SQL-specific optimizations

            # Remove SQL verbosity
            sql_verbosity = [
                r"Can you (please )?(write|create|generate) (a )?(SQL )?query",
                r"I need (a )?(SQL )?query to",
                r"(Write|Create|Generate) (a )?(SQL )?query",
                r"(using|with) SQL",
                r"in SQL syntax",
                r"using (the )?(following|this) table",
            ]

            optimized = query
            for pattern in sql_verbosity:
                optimized = re.sub(pattern, "", optimized, flags=re.IGNORECASE)

            return optimized.strip()

        elif domain == "data_science":
            # Data science specific optimizations

            # Remove data science verbosity
            ds_verbosity = [
                r"Can you (please )?(analyze|examine|study)",
                r"I need (to|you to) (analyze|examine)",
                r"(Perform|Do|Conduct) (a|an) (analysis|examination)",
                r"(using|with) (machine learning|ML|AI|statistical methods)",
                r"with (the )?(dataset|data)",
            ]

            optimized = query
            for pattern in ds_verbosity:
                optimized = re.sub(pattern, "", optimized, flags=re.IGNORECASE)

            return optimized.strip()

        # Default - no domain-specific optimization
        return query

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

        # Calculate message relevance scores for semantic pruning
        last_message = non_system_messages[-1] if non_system_messages else None

        # Keep the most recent message and score others for retention
        if last_message:
            # Simplified relevance scoring (more sophisticated would use embeddings)
            last_message_content = getattr(last_message, "content", "")

            # Score each message by their relevance to the last message
            message_scores = []
            for i, msg in enumerate(non_system_messages[:-1]):  # All but the last message
                msg_content = getattr(msg, "content", "")

                # Calculate text overlap-based relevance
                # (In a production system, use embeddings for better semantic similarity)
                words_last = set(last_message_content.lower().split())
                words_this = set(msg_content.lower().split())

                if not words_this or not words_last:
                    relevance = 0.0
                else:
                    overlap = len(words_this.intersection(words_last))
                    relevance = overlap / max(len(words_this), len(words_last))

                # Recency bonus - more recent messages get a bonus
                recency_bonus = (i + 1) / len(non_system_messages)

                # Final score combines relevance and recency
                final_score = (0.7 * relevance) + (0.3 * recency_bonus)
                message_scores.append((i, final_score))

            # Sort by score (highest first)
            message_scores.sort(key=lambda x: x[1], reverse=True)

            # Select top-scoring messages up to max_messages - 1 (save room for the last message)
            selected_indices = [idx for idx, _ in message_scores[: max_messages - 1]]
            selected_indices.sort()  # Keep original order

            # Construct final pruned list
            pruned_non_system = [non_system_messages[i] for i in selected_indices]
            pruned_non_system.append(non_system_messages[-1])  # Always keep the most recent message

            # Combine system messages with pruned non-system messages
            return system_messages + pruned_non_system

        # Fallback to the original pruning method
        recent_messages = non_system_messages[-max_messages:]
        return system_messages + recent_messages

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


# Add a function to record user feedback
def record_query_feedback(query: str, rating: int, comments: Optional[str] = None) -> None:
    """Record user feedback on response quality for a query.

    Args:
        query: The original user query
        rating: User rating (1-5)
        comments: Optional user comments
    """
    # Create performance monitor and record feedback
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    monitor = PerformanceMonitor(log_file=str(logs_dir / "model_performance.json"))
    monitor.record_feedback(query, rating, comments)
