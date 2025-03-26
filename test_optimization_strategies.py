import json
from datetime import datetime

# Import the optimization components
from querymancer.optimizations import (
    DynamicComplexityRouter,
    TokenOptimizationPipeline,
    optimize_query_execution,
)
from langchain_core.messages import HumanMessage, SystemMessage

# Test queries with varying complexity
TEST_QUERIES = [
    # Simple queries (should route to Groq)
    "List all tables in the database",
    "What's the schema of the users table?",
    "Count the number of orders in the database",
    # Complex queries (should route to SambaNova)
    "Analyze the purchase patterns across different customer segments and identify trends in the last 12 months",
    "Create a cohort analysis of customer retention rates by signup month and calculate the month-over-month change",
    "Find correlations between customer demographics and product categories with the highest profit margins",
]


def print_divider():
    print("\n" + "=" * 80 + "\n")


def test_complexity_router():
    """Test the DynamicComplexityRouter's ability to analyze query complexity."""
    print_divider()
    print("TESTING DYNAMIC COMPLEXITY ROUTER")
    print_divider()

    router = DynamicComplexityRouter()

    results = []
    for i, query in enumerate(TEST_QUERIES):
        # Analyze query complexity
        score, dimensions = router.analyze_complexity(query)

        # Determine which model would be selected
        provider = "SambaNova" if router.should_use_complex_model(query) else "Groq"

        result = {
            "query": query,
            "complexity_score": score,
            "dimensions": dimensions,
            "selected_provider": provider,
            "threshold": router.threshold,
        }
        results.append(result)

        # Print the results
        print(f"Query {i+1}: {query[:50]}{'...' if len(query) > 50 else ''}")
        print(f"  Complexity Score: {score:.2f}")
        print("  Dimensions:")
        for dim, value in dimensions.items():
            print(f"    - {dim}: {value:.2f}")
        print(f"  Selected Provider: {provider}")
        print(
            f"  Routed to {'Complex Model' if router.should_use_complex_model(query) else 'Simple Model'}"
        )
        print()

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"complexity_router_test_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Full results saved to: complexity_router_test_{timestamp}.json\n")
    return results


def test_token_optimization():
    """Test the TokenOptimizationPipeline's ability to refine queries."""
    print_divider()
    print("TESTING TOKEN OPTIMIZATION PIPELINE")
    print_divider()

    optimizer = TokenOptimizationPipeline()

    # Add some verbose queries with filler phrases
    verbose_queries = [
        "Could you please tell me what are all the tables in the database?",
        "I was wondering if you could provide me with information about the schema of the users table, thanks in advance!",
        "I would really like to know the count of orders in the database if that's not too much trouble",
        "Please analyze the very complex and detailed purchase patterns across different customer segments",
    ]

    results = []
    for i, query in enumerate(verbose_queries):
        # Optimize the query
        optimized = optimizer.refine_query(query)

        # Calculate token reduction
        original_tokens = len(query.split())
        optimized_tokens = len(optimized.split())
        token_reduction = original_tokens - optimized_tokens
        reduction_percentage = (
            (token_reduction / original_tokens) * 100 if original_tokens > 0 else 0
        )

        result = {
            "original_query": query,
            "optimized_query": optimized,
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "token_reduction": token_reduction,
            "reduction_percentage": reduction_percentage,
        }
        results.append(result)

        # Print the results
        print(f"Original Query: {query}")
        print(f"Optimized Query: {optimized}")
        print(f"Token Reduction: {token_reduction} tokens ({reduction_percentage:.1f}%)")
        print()

    # Test model parameter optimization
    print("MODEL PARAMETER OPTIMIZATION:")

    # Test for different complexity levels and providers
    test_cases = [
        {"complexity": 0.2, "provider": "groq"},
        {"complexity": 0.7, "provider": "groq"},
        {"complexity": 0.2, "provider": "sambanova"},
        {"complexity": 0.7, "provider": "sambanova"},
    ]

    for case in test_cases:
        params = optimizer.optimize_model_params(case["complexity"], case["provider"])
        print(f"Complexity: {case['complexity']:.1f}, Provider: {case['provider']}")
        print(f"  Optimized parameters: {params}")

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"token_optimization_test_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nFull results saved to: token_optimization_test_{timestamp}.json\n")
    return results


def test_full_pipeline():
    """Test the complete optimization pipeline."""
    print_divider()
    print("TESTING FULL OPTIMIZATION PIPELINE")
    print_divider()

    # Create a test conversation history
    history = [
        SystemMessage(content="You are a helpful SQL assistant."),
        HumanMessage(content="What database tables do you have?"),
        SystemMessage(content="I'll help you explore the database."),
    ]

    # Test with each query
    for i, query in enumerate(TEST_QUERIES):
        print(f"Query {i+1}: {query}")

        # Execute the full optimization pipeline
        optimized_query, model, optimized_history, model_params = optimize_query_execution(
            query, history
        )

        # Print results
        print(f"  Original Query: {query}")
        print(f"  Optimized Query: {optimized_query}")
        print(f"  Selected Model Provider: {model.__class__.__name__}")
        print(f"  Original History Length: {len(history)}")
        print(f"  Optimized History Length: {len(optimized_history)}")
        print(f"  Optimized Parameters: {model_params}")
        print()


if __name__ == "__main__":
    print("TESTING QUERYMANCER OPTIMIZATION STRATEGIES")
    print("===========================================")

    try:
        # Test each component
        test_complexity_router()
        test_token_optimization()
        test_full_pipeline()

        print("All tests completed successfully!")
    except Exception as e:
        print(f"Error during testing: {str(e)}")
