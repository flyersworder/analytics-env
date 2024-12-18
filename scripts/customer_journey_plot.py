import plotly.graph_objects as go
import pandas as pd
from datetime import timedelta


def create_customer_journey(data, key_events=None, padding_days=10):
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

    # Dynamic vertical positioning to avoid overlap
    y_positions = []
    last_time = None
    last_position = 0

    for i, time in enumerate(df["Time"]):
        if last_time and (time - last_time).days <= 3:
            last_position += 50
        else:
            last_position = 50 if i % 2 == 0 else -50
        y_positions.append(last_position)
        last_time = time

    df["Y_Position"] = y_positions
    df["Is_Key_Event"] = df["Activity"].isin(key_events) if key_events else False

    # Create the figure
    fig = go.Figure()

    # Add dashed placeholders for inactive periods
    placeholder_df = pd.date_range(start=start_date, end=end_date, freq="D")
    for i in range(len(placeholder_df) - 1):
        fig.add_trace(
            go.Scatter(
                x=[placeholder_df[i], placeholder_df[i + 1]],
                y=[0, 0],
                mode="lines",
                line=dict(color="lightgray", dash="dash"),
                hoverinfo="none",
            )
        )

    # Add markers for actual events
    for _, row in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["Time"]],
                y=[0],
                mode="markers",
                marker=dict(
                    size=18 if row["Is_Key_Event"] else 14,
                    color="red" if row["Is_Key_Event"] else "#1f77b4",
                    line=dict(width=1.5, color="black"),
                ),
                hovertext=f"{row['Activity']}: {row['Description']}",
                hoverinfo="text",
            )
        )

    # Add annotations
    for _, row in df.iterrows():
        fig.add_annotation(
            x=row["Time"],
            y=0,
            text=f"<b>{row['Activity']}</b><br>{row['Description']}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="#999999",
            ax=0,
            ay=row["Y_Position"],
            font=dict(size=12, color="#333333"),
            align="left",
            bgcolor="#f9f9f9",
            bordercolor="#cccccc",
            borderwidth=1,
            borderpad=4,
        )

    # Style adjustments
    fig.update_layout(
        title="Customer Journey Timeline (Automated Dates)",
        xaxis=dict(
            title="Time",
            range=[start_date, end_date],
            showgrid=False,
            showline=True,
            linecolor="#cccccc",
            tickformat="%Y-%m-%d",
            tickfont=dict(size=12, color="#333333"),
        ),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor="#ffffff",
        height=700,
        showlegend=False,
    )

    return fig


test_cases = [
    {
        "title": "Sparse Events",
        "data": {
            "Activity": ["Download Data Sheet", "BuyButton", "Attend Webinar"],
            "Description": [
                "Data Sheet: Product A",
                "Purchase Button Clicked",
                "Webinar: AI Trends",
            ],
            "Time": ["2024-01-01", "2024-02-10", "2024-02-25"],
        },
        "key_events": ["BuyButton"],
    },
    {
        "title": "Intense Events",
        "data": {
            "Activity": ["Event 1", "Event 2", "BuyButton", "Event 4"],
            "Description": [
                "Description 1",
                "Description 2",
                "Purchase Made",
                "Description 4",
            ],
            "Time": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        },
        "key_events": ["BuyButton"],
    },
    {
        "title": "Single Event",
        "data": {
            "Activity": ["Single Event"],
            "Description": ["A single event occurred"],
            "Time": ["2024-01-15"],
        },
        "key_events": ["Single Event"],
    },
    {
        "title": "Evenly Distributed Events",
        "data": {
            "Activity": ["Event A", "Event B", "Event C", "Event D"],
            "Description": [
                "Description A",
                "Description B",
                "Description C",
                "Description D",
            ],
            "Time": ["2024-01-01", "2024-01-10", "2024-01-20", "2024-01-30"],
        },
        "key_events": [],
    },
    {
        "title": "All Key Events",
        "data": {
            "Activity": ["KeyEvent1", "KeyEvent2", "KeyEvent3"],
            "Description": [
                "Key description 1",
                "Key description 2",
                "Key description 3",
            ],
            "Time": ["2024-01-05", "2024-01-15", "2024-01-25"],
        },
        "key_events": ["KeyEvent1", "KeyEvent2", "KeyEvent3"],
    },
    {
        "title": "Very Dense Events",
        "data": {
            "Activity": ["Event X1", "Event X2", "Event X3", "Event X4"],
            "Description": [
                "Dense event 1",
                "Dense event 2",
                "Dense event 3",
                "Dense event 4",
            ],
            "Time": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"],
        },
        "key_events": [],
    },
    {
        "title": "Long Timeline",
        "data": {
            "Activity": ["Event L1", "Event L2", "Event L3", "Event L4"],
            "Description": [
                "Long timeline 1",
                "Long timeline 2",
                "Long timeline 3",
                "Long timeline 4",
            ],
            "Time": ["2024-01-01", "2024-03-01", "2024-06-01", "2024-12-01"],
        },
        "key_events": [],
    },
    {
        "title": "No Events",
        "data": {"Activity": [], "Description": [], "Time": []},
        "key_events": [],
    },
]

# Rerun test scenarios
figures = []
for case in test_cases:
    fig = create_customer_journey(case["data"], key_events=case["key_events"])
    fig.update_layout(title=case["title"])
    figures.append(fig)

# Display results for all test cases
for fig in figures:
    fig.show()
