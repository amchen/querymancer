"""
Querymancer Performance Dashboard

This dashboard visualizes the performance metrics and calibration history
collected by the Querymancer optimization system.
"""

import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Querymancer Performance Dashboard",
    page_icon="📊",
    layout="wide",
)

# Dashboard title
st.title("Querymancer Performance Dashboard")
st.markdown(
    """
    This dashboard visualizes the performance metrics and calibration history
    collected by the Querymancer optimization system.
    """
)


def load_performance_data():
    """Load performance data from the logs directory."""
    logs_dir = Path("logs")
    performance_file = logs_dir / "model_performance.json"

    if not performance_file.exists():
        return None

    try:
        with open(performance_file, "r") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, FileNotFoundError):
        return None


# Load performance data
performance_data = load_performance_data()

if performance_data is None:
    st.warning("No performance data found. Run some queries with Querymancer to generate data.")
    st.stop()

# Create tabs for different dashboard sections
tab1, tab2, tab3, tab4 = st.tabs(
    ["Model Selection", "User Feedback", "Complexity Analysis", "Threshold Calibration"]
)

# Tab 1: Model Selection
with tab1:
    st.header("Model Selection Analysis")

    # Extract selection data
    if "selections" in performance_data and performance_data["selections"]:
        selections_df = pd.DataFrame(performance_data["selections"])

        # Convert timestamps to datetime
        selections_df["timestamp"] = pd.to_datetime(selections_df["timestamp"])
        selections_df = selections_df.sort_values("timestamp")

        # Model distribution
        st.subheader("Model Selection Distribution")
        model_counts = selections_df["selected_model"].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            model_counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#55A868"])
            ax.set_xlabel("Model")
            ax.set_ylabel("Number of Queries")
            ax.set_title("Distribution of Queries by Model")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            model_counts.plot(kind="pie", ax=ax, autopct="%1.1f%%", colors=["#4C72B0", "#55A868"])
            ax.set_title("Percentage of Queries by Model")
            ax.set_ylabel("")
            st.pyplot(fig)

        # Model selection over time
        st.subheader("Model Selection Over Time")

        # Group by day and model
        selections_df["date"] = selections_df["timestamp"].dt.date
        time_series = selections_df.groupby(["date", "selected_model"]).size().unstack().fillna(0)

        if not time_series.empty and time_series.shape[0] > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            time_series.plot(kind="line", ax=ax, marker="o")
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Queries")
            ax.set_title("Model Selection Over Time")
            ax.legend(title="Model")
            st.pyplot(fig)
        else:
            st.info("Not enough time series data to plot model selection over time.")

        # Complexity distribution by model
        st.subheader("Complexity Score Distribution by Model")

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(
            data=selections_df, x="complexity", hue="selected_model", kde=True, bins=20, ax=ax
        )
        ax.set_xlabel("Complexity Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Complexity Scores by Selected Model")
        st.pyplot(fig)

        # Display recent selections
        st.subheader("Recent Model Selections")
        recent_selections = selections_df.sort_values("timestamp", ascending=False).head(10)
        recent_selections["timestamp"] = recent_selections["timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        st.dataframe(
            recent_selections[["timestamp", "query", "complexity", "selected_model"]],
            use_container_width=True,
        )
    else:
        st.info("No model selection data available yet.")

# Tab 2: User Feedback
with tab2:
    st.header("User Feedback Analysis")

    # Extract feedback data
    if "feedback" in performance_data and performance_data["feedback"]:
        feedback_df = pd.DataFrame(performance_data["feedback"])

        # Convert timestamps to datetime
        feedback_df["timestamp"] = pd.to_datetime(feedback_df["timestamp"])
        feedback_df = feedback_df.sort_values("timestamp")

        # Rating distribution
        st.subheader("Rating Distribution")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            rating_counts = feedback_df["rating"].value_counts().sort_index()
            rating_counts.plot(
                kind="bar", ax=ax, color=sns.color_palette("YlGnBu", len(rating_counts))
            )
            ax.set_xlabel("Rating")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of User Ratings")
            ax.set_xticks(range(len(rating_counts)))
            ax.set_xticklabels(rating_counts.index)
            st.pyplot(fig)

        with col2:
            # Average rating over time
            if feedback_df.shape[0] > 1:
                feedback_df["date"] = feedback_df["timestamp"].dt.date
                daily_avg = feedback_df.groupby("date")["rating"].mean()

                fig, ax = plt.subplots(figsize=(8, 6))
                daily_avg.plot(kind="line", ax=ax, marker="o", color="#4C72B0")
                ax.set_xlabel("Date")
                ax.set_ylabel("Average Rating")
                ax.set_title("Average Rating Over Time")
                ax.set_ylim(0, 5.5)
                ax.axhline(
                    y=daily_avg.mean(),
                    color="r",
                    linestyle="--",
                    label=f"Overall Avg: {daily_avg.mean():.2f}",
                )
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("Not enough data to plot ratings over time.")

        # Link feedback to model selections
        st.subheader("Ratings by Model")

        # Attempt to join feedback with selections
        if "selections" in performance_data and performance_data["selections"]:
            selections_df = pd.DataFrame(performance_data["selections"])

            # Create a mapping from query to model
            query_to_model = {
                row["query"]: row["selected_model"] for _, row in selections_df.iterrows()
            }

            # Add model information to feedback
            feedback_df["model"] = feedback_df["query"].map(query_to_model)

            # Filter out rows with no model information
            feedback_with_model = feedback_df.dropna(subset=["model"])

            if not feedback_with_model.empty:
                # Calculate average rating by model
                model_ratings = feedback_with_model.groupby("model")["rating"].agg(
                    ["mean", "count", "std"]
                )
                model_ratings = model_ratings.rename(
                    columns={
                        "mean": "Average Rating",
                        "count": "Number of Ratings",
                        "std": "Standard Deviation",
                    }
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.dataframe(model_ratings.round(2), use_container_width=True)

                with col2:
                    if len(model_ratings) > 1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        model_ratings["Average Rating"].plot(
                            kind="bar",
                            ax=ax,
                            yerr=model_ratings["Standard Deviation"],
                            capsize=10,
                            color=["#4C72B0", "#55A868"],
                        )
                        ax.set_xlabel("Model")
                        ax.set_ylabel("Average Rating")
                        ax.set_title("Average Rating by Model")
                        ax.set_ylim(0, 5.5)
                        st.pyplot(fig)
                    else:
                        st.info("Need ratings for multiple models to compare.")
            else:
                st.info("No feedback data could be linked to model selections.")
        else:
            st.info("No model selection data available to link with feedback.")

        # Display recent feedback
        st.subheader("Recent User Feedback")
        recent_feedback = feedback_df.sort_values("timestamp", ascending=False).head(10)
        recent_feedback["timestamp"] = recent_feedback["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Truncate query text for display
        recent_feedback["query_short"] = recent_feedback["query"].apply(
            lambda x: x[:100] + "..." if len(x) > 100 else x
        )

        st.dataframe(
            recent_feedback[["timestamp", "query_short", "rating", "comments"]],
            use_container_width=True,
        )
    else:
        st.info("No user feedback data available yet.")

# Tab 3: Complexity Analysis
with tab3:
    st.header("Complexity Analysis")

    if "selections" in performance_data and performance_data["selections"]:
        selections_df = pd.DataFrame(performance_data["selections"])

        # Convert timestamps to datetime
        selections_df["timestamp"] = pd.to_datetime(selections_df["timestamp"])

        # Complexity distribution
        st.subheader("Complexity Score Distribution")

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data=selections_df, x="complexity", kde=True, bins=20, ax=ax)
        ax.set_xlabel("Complexity Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Query Complexity Scores")

        # Add vertical line for typical threshold
        threshold = 0.25  # Default threshold
        ax.axvline(x=threshold, color="r", linestyle="--", label=f"Typical Threshold: {threshold}")
        ax.legend()

        st.pyplot(fig)

        # Complexity vs. query length
        st.subheader("Complexity vs. Query Length")

        # Calculate query length
        selections_df["query_length"] = selections_df["query"].apply(len)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.scatterplot(
            data=selections_df,
            x="query_length",
            y="complexity",
            hue="selected_model",
            alpha=0.7,
            ax=ax,
        )
        ax.set_xlabel("Query Length (characters)")
        ax.set_ylabel("Complexity Score")
        ax.set_title("Relationship Between Query Length and Complexity")

        # Add horizontal line for typical threshold
        ax.axhline(y=threshold, color="r", linestyle="--", label=f"Typical Threshold: {threshold}")
        ax.legend()

        st.pyplot(fig)

        # Complexity trends over time
        st.subheader("Complexity Trends Over Time")

        selections_df["date"] = selections_df["timestamp"].dt.date
        daily_complexity = selections_df.groupby("date")["complexity"].mean()

        if len(daily_complexity) > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            daily_complexity.plot(kind="line", marker="o", ax=ax)
            ax.set_xlabel("Date")
            ax.set_ylabel("Average Complexity Score")
            ax.set_title("Average Query Complexity Over Time")
            ax.axhline(
                y=threshold, color="r", linestyle="--", label=f"Typical Threshold: {threshold}"
            )
            ax.legend()

            st.pyplot(fig)
        else:
            st.info("Not enough data to plot complexity trends over time.")
    else:
        st.info("No complexity data available yet.")

# Tab 4: Threshold Calibration
with tab4:
    st.header("Threshold Calibration")

    # Check if we have calibration history data
    calibration_file = Path("logs") / "calibration_history.json"

    if calibration_file.exists():
        try:
            with open(calibration_file, "r") as f:
                calibration_data = json.load(f)

            if calibration_data:
                calibration_df = pd.DataFrame(calibration_data)
                calibration_df["timestamp"] = pd.to_datetime(calibration_df["timestamp"])
                calibration_df = calibration_df.sort_values("timestamp")

                # Threshold changes over time
                st.subheader("Threshold Changes Over Time")

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(
                    calibration_df["timestamp"],
                    calibration_df["old_threshold"],
                    marker="o",
                    linestyle="--",
                    label="Previous Threshold",
                )
                ax.plot(
                    calibration_df["timestamp"],
                    calibration_df["new_threshold"],
                    marker="o",
                    label="New Threshold",
                )
                ax.set_xlabel("Date")
                ax.set_ylabel("Complexity Threshold")
                ax.set_title("Threshold Calibration History")
                ax.legend()

                st.pyplot(fig)

                # Display calibration events
                st.subheader("Calibration Events")

                display_df = calibration_df.copy()
                display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
                display_df["threshold_change"] = (
                    display_df["new_threshold"] - display_df["old_threshold"]
                )
                display_df["threshold_change_pct"] = (
                    display_df["threshold_change"] / display_df["old_threshold"]
                ) * 100

                st.dataframe(
                    display_df[
                        [
                            "timestamp",
                            "old_threshold",
                            "new_threshold",
                            "threshold_change",
                            "threshold_change_pct",
                            "groq_avg_rating",
                            "sambanova_avg_rating",
                        ]
                    ].round(4),
                    use_container_width=True,
                )
            else:
                st.info("Calibration history file exists but contains no data.")
        except (json.JSONDecodeError, FileNotFoundError):
            st.warning("Error reading calibration history file.")
    else:
        # If we don't have real calibration data, show a simulation
        st.info("No threshold calibration history available yet. Showing a simulation.")

        # Create simulated calibration data
        np.random.seed(42)
        dates = pd.date_range(start="2025-01-01", periods=30, freq="D")

        initial_threshold = 0.25
        thresholds = [initial_threshold]

        for i in range(1, 30):
            # Simulate small adjustments
            adjustment = np.random.normal(0, 0.01)
            # Ensure we stay within reasonable bounds
            new_threshold = max(0.1, min(0.4, thresholds[-1] + adjustment))
            thresholds.append(new_threshold)

        # Create a dataframe with the simulated data
        sim_df = pd.DataFrame({"date": dates, "threshold": thresholds})

        # Plot the simulated data
        fig, ax = plt.subplots(figsize=(12, 6))
        sim_df.plot(x="date", y="threshold", ax=ax, marker="o")
        ax.set_xlabel("Date")
        ax.set_ylabel("Complexity Threshold")
        ax.set_title("Simulated Threshold Calibration Over Time")
        ax.axhline(y=initial_threshold, color="r", linestyle="--", label="Initial Threshold")
        ax.legend()

        st.pyplot(fig)

        st.markdown("""
        This simulation shows how the threshold might evolve over time as the system 
        learns from user feedback and model performance. The actual calibration will 
        be based on real performance data once enough queries and feedback are collected.
        """)

# Add a section for manual feedback collection
st.header("Submit Feedback")
st.markdown("""
Use this form to submit feedback on a recent query. This helps improve the model selection process.
""")

with st.form("feedback_form"):
    query = st.text_area("Query", placeholder="Enter the query you want to provide feedback on")
    rating = st.slider("Rating", min_value=1, max_value=5, value=3, help="1 = Poor, 5 = Excellent")
    comments = st.text_area("Comments", placeholder="Optional comments about the response quality")

    submit_button = st.form_submit_button("Submit Feedback")

    if submit_button:
        if query:
            # Import the feedback function
            try:
                from querymancer.optimizations import record_query_feedback

                # Record the feedback
                record_query_feedback(query, rating, comments)

                st.success("Feedback submitted successfully!")
                st.info(
                    "The dashboard will update with your feedback the next time it's refreshed."
                )
            except ImportError:
                st.error(
                    "Could not import the feedback recording function. Make sure you're running this from the Querymancer directory."
                )
        else:
            st.warning("Please enter a query to provide feedback on.")

# Add information about how to run the dashboard
st.sidebar.title("About")
st.sidebar.info("""
This dashboard visualizes the performance metrics and calibration history
collected by the Querymancer optimization system.

To run this dashboard:
```
streamlit run dashboard.py
```

Data is loaded from the `logs` directory.
""")

# Add refresh button
if st.sidebar.button("Refresh Data"):
    st.experimental_rerun()

# Display timestamp
st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
