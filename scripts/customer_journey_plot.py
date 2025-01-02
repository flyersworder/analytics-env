import plotly.graph_objects as go
import pandas as pd
from datetime import timedelta


def wrap_text(text, max_chars=40):
    """Wrap text into multiple lines with a given character limit."""
    words = text.split()
    lines, current_line = [], []

    for word in words:
        if (
            sum(len(w) for w in current_line) + len(current_line) + len(word)
            <= max_chars
        ):
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
    lines.append(" ".join(current_line))
    return "<br>".join(lines)


def create_customer_journey_with_ft_style(
    data, key_events=None, padding_days=10, max_chars=40
):
    # Helper function to wrap text
    def wrap_text(text, max_chars):
        words = text.split()
        lines, current_line = [], []
        for word in words:
            if (
                sum(len(w) for w in current_line) + len(current_line) + len(word)
                <= max_chars
            ):
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
        lines.append(" ".join(current_line))
        return "<br>".join(lines)

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    if df.empty or df["Time"].isna().all():
        # Handle empty or invalid data
        fig = go.Figure()
        fig.add_annotation(
            text="No events to display",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="red"),
        )
        fig.update_layout(
            title="Customer Journey Timeline (No Events)",
            plot_bgcolor="#ffffff",
            height=400,
        )
        return fig

    # Sort events by time
    df = df.dropna(subset=["Time"]).sort_values(by="Time").reset_index(drop=True)

    # Automatically calculate start and end dates with padding
    start_date = df["Time"].min() - timedelta(days=padding_days)
    end_date = df["Time"].max() + timedelta(days=padding_days)

    # Define unique journey stages and assign y-axis positions
    unique_journeys = df["Journey"].unique()
    journey_positions = {journey: i for i, journey in enumerate(unique_journeys)}
    df["Journey_Position"] = df["Journey"].map(journey_positions)

    # Ensure the 'Is_Key_Event' column is created properly
    df["Is_Key_Event"] = df["Activity"].isin(key_events) if key_events else False

    # FT-Style Color Palette for Journey Stages
    ft_colors = ["#e0d5c6", "#b6cce5", "#a4c3a8", "#f1a6ab", "#dfd3e6", "#f4e9b6"]
    journey_colors = {
        journey: ft_colors[i % len(ft_colors)]
        for i, journey in enumerate(unique_journeys)
    }

    # Create the figure
    fig = go.Figure()

    # Add journey stage bands
    for journey, position in journey_positions.items():
        fig.add_shape(
            type="rect",
            x0=start_date,
            x1=end_date,
            y0=position - 0.5,
            y1=position + 0.5,
            fillcolor=journey_colors[journey],
            opacity=0.4,
            layer="below",
            line_width=0,
        )

    # Add markers for actual events
    for _, row in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["Time"]],
                y=[row["Journey_Position"]],
                mode="markers",
                marker=dict(
                    size=18 if row["Is_Key_Event"] else 14,
                    color="#5a5a5a" if not row["Is_Key_Event"] else "#d62728",
                    line=dict(width=1.5, color="black"),
                ),
                hovertext=f"{row['Activity']}: {row['Description']}",
                hoverinfo="text",
                showlegend=False,  # Disable legend for individual events
            )
        )

    # Add annotations with wrapped text
    for _, row in df.iterrows():
        wrapped_text = (
            f"<b>{row['Activity']}</b><br>{wrap_text(row['Description'], max_chars)}"
        )
        fig.add_annotation(
            x=row["Time"],
            y=row["Journey_Position"],
            text=wrapped_text,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="#999999",
            ax=0,
            ay=-50,  # Adjust position slightly above the marker
            font=dict(size=12, color="#333333"),
            align="left",
            bgcolor="#f9f9f9",
            bordercolor="#cccccc",
            borderwidth=1,
            borderpad=4,
        )

    # Add a legend for journey stages
    for journey, color in journey_colors.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                name=journey,
            )
        )

    # Style adjustments for FT aesthetic
    fig.update_layout(
        title="Customer Journey Timeline (FT Style)",
        xaxis=dict(
            title="Time",
            range=[start_date, end_date],
            showgrid=False,
            showline=True,
            linecolor="#999999",
            tickformat="%Y-%m-%d",
            tickfont=dict(size=12, color="#333333"),
        ),
        yaxis=dict(
            title="Journey Stages",
            tickmode="array",
            tickvals=list(journey_positions.values()),
            ticktext=list(journey_positions.keys()),
            showgrid=False,
            zeroline=False,
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor="#ffffff",
        height=700,
        showlegend=True,
    )

    return fig


# Test case with journeys
test_data = {
    "Activity": [
        "Download Data Sheet",
        "BuyButton",
        "Attend Webinar",
        "Prototype Testing",
    ],
    "Description": [
        "Downloaded data sheet for Product A",
        "Clicked the purchase button",
        "Attended an AI trends webinar",
        "Completed prototype testing",
    ],
    "Time": ["2024-01-01", "2024-02-10", "2024-02-25", "2024-03-15"],
    "Journey": ["Awareness", "Purchase", "Consider", "Prototype"],
}

# Generate the figure
fig = create_customer_journey_with_ft_style(test_data, key_events=["BuyButton"])
fig.show()
