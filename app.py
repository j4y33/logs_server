# streamlit_sensor_data.py
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sqlalchemy
from sqlalchemy.pool import QueuePool
import sqlalchemy.sql.expression as sql
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger()

# Check if running in Streamlit Cloud or locally
def is_streamlit_cloud():
    return 'STREAMLIT_SHARING_MODE' in os.environ or 'STREAMLIT_RUN_PATH' in os.environ

def get_user_ids(engine):
    """
    Get a list of all user IDs from the user_profiles table
    
    Args:
        engine: SQLAlchemy database engine
        
    Returns:
        List of user IDs as strings
    """
    try:
        # Using parameterized query for safety
        query = sql.text("SELECT DISTINCT user_id FROM user_profile ORDER BY user_id")
        with engine.connect() as conn:
            result = conn.execute(query)
            return [str(row[0]) for row in result]  # Convert UUID to string
    except Exception as e:
        logger.error(f"Error fetching user IDs: {e}")
        return []

def plot_sensor_data(
    engine,
    user_id,
    start_date,
    end_date,
    granularity='hour'
):
    """
    Plot sensor data over time for a specific user and date range.
    
    Args:
        engine: SQLAlchemy database engine
        user_id: User ID to analyze
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        granularity: Granularity of the bar plot ('minute', 'hour', 'day')
    
    Returns:
        plotly figure for interactive visualization
    """
    # Convert dates for display and queries
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    # Format for SQL
    start_str = start_ts.strftime('%Y-%m-%d')
    end_str = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Full timestamp strings for plot range
    start_ts_str = start_ts.isoformat()
    end_ts_str = end_ts.isoformat()
    
    # Sensor tables to query
    tables = {
        'accelerometer_data': 'Accelerometer',
        'gyroscope_data': 'Gyroscope',
        'magnetometer_data': 'Magnetometer',
        'locations': 'Location'
    }
    
    # Create plotly figure with subplots
    fig = make_subplots(
        rows=len(tables), 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=list(tables.values())
    )
    
    # Colors for each sensor
    colors = ['blue', 'green', 'red', 'purple']
    
    # For each sensor table
    for i, (table, label) in enumerate(tables.items()):
        # Using parameterized query instead of f-string for security
        query = sql.text(f"""
            SELECT timestamp 
            FROM {table}
            WHERE user_id = :user_id
            AND timestamp >= :start_date
            AND timestamp < :end_date
            ORDER BY timestamp
        """)
        
        try:
            logger.info(f"Fetching data from {table}...")
            
            # Execute with parameters
            with engine.connect() as conn:
                result = conn.execute(
                    query, 
                    {"user_id": str(user_id), "start_date": start_str, "end_date": end_str}
                )
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df.empty:
                # Convert to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.info(f"Found {len(df)} records in {table}")
                
                # Always use hourly granularity
                df['time_bucket'] = df['timestamp'].dt.floor('h')
                
                # Group and count
                time_counts = df.groupby('time_bucket').size().reset_index()
                time_counts.columns = ['time_bucket', 'count']
                
                # Add bar plot
                fig.add_trace(
                    go.Bar(
                        x=time_counts['time_bucket'],
                        y=time_counts['count'],
                        marker_color=colors[i],
                        opacity=0.8,
                        name=f"{label}",
                        hovertemplate='%{x}<br>Count: %{y}<extra></extra>'
                    ),
                    row=i+1, 
                    col=1
                )
                
                # Add text annotation with record count
                fig.add_annotation(
                    x=0.01, 
                    y=0.9,
                    xref="paper",
                    yref="paper",
                    text=f"Total: {len(df):,} records",
                    showarrow=False,
                    font=dict(color="white"),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="white",
                    borderwidth=1,
                    borderpad=4,
                    opacity=0.8,
                    row=i+1,
                    col=1
                )
                
                # Hide y-axis as it's not meaningful
                fig.update_yaxes(
                    title_text="Count",
                    row=i+1, 
                    col=1
                )
                
            else:
                logger.warning(f"No data found for {table}")
                # Add empty trace with message
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    text="No data available",
                    showarrow=False,
                    font=dict(color="gray", size=14),
                    row=i+1,
                    col=1
                )
                    
        except Exception as e:
            logger.error(f"Error processing {table}: {e}")
            # Add error annotation
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                text=f"Error: {str(e)}",
                showarrow=False,
                font=dict(color="red", size=14),
                row=i+1,
                col=1
            )
    
    # Update layout for better visualization
    fig.update_layout(
        height=800,
        title=f'Hourly Sensor Data ({start_date} to {end_date})',
        hovermode='closest',
        template="plotly_dark",
        dragmode='zoom',  # Default to zoom mode
        margin=dict(t=80, l=50, r=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add grid lines and set date format for all x-axes
    for i in range(1, len(tables) + 1):
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)',
            type='date',  # Explicitly set as date
            tickformat='%Y-%m-%d %H:%M',  # Format the tick labels
            row=i,
            col=1,
            range=[start_ts_str, end_ts_str]  # Use full timestamp strings
        )
    
    # Add simple time navigation buttons to bottom x-axis only (last subplot)
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=3, label="3d", step="day", stepmode="backward"),
                dict(step="all", label="All")
            ])
        ),
        row=len(tables),  # Only add to the last row
        col=1
    )
    
    return fig

def main():
    st.set_page_config(page_title="Sensor Data Viewer", layout="wide")
    
    st.title("Sensor Data Visualization Tool")
    st.write("Visualize hourly sensor data across different users")
    
    # Check for database connection based on environment
    db_connection_string = None
    
    if is_streamlit_cloud():
        # When in Streamlit Cloud, use secrets
        if 'connections' in st.secrets and 'db' in st.secrets.connections:
            db_connection_string = st.secrets.connections.db.url
        else:
            st.error("Database connection not configured in Streamlit Cloud secrets.")
            st.info("Please configure secrets in the Streamlit Cloud dashboard.")
            return
    else:
        # Local development - check for .env file first, then secrets
        try:
            from dotenv import load_dotenv
            load_dotenv()
            db_connection_string = os.environ.get("DB_CONNECTION_STAGING")
            
            # If not in .env file, try local secrets
            if not db_connection_string and 'connections' in st.secrets and 'db' in st.secrets.connections:
                db_connection_string = st.secrets.connections.db.url
                
            if not db_connection_string:
                st.error("Database connection string not found.")
                st.info("For local development, add a .env file with DB_CONNECTION_STAGING or a .streamlit/secrets.toml file.")
                return
        except ImportError:
            st.error("dotenv package not installed for local development. Using only secrets.")
            if 'connections' in st.secrets and 'db' in st.secrets.connections:
                db_connection_string = st.secrets.connections.db.url
            else:
                st.error("Database connection string not found in secrets.")
                return
    
    try:
        # Create database engine with connection pooling
        engine = sqlalchemy.create_engine(
            db_connection_string,
            poolclass=QueuePool,
            pool_size=3,
            max_overflow=5,
            pool_timeout=30,
            pool_recycle=1800  # Recycle connections after 30 minutes
        )
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(sql.text("SELECT 1"))
            st.success("Connected to database successfully!")
        
        # Get list of user IDs
        user_ids = get_user_ids(engine)
        
        if not user_ids:
            st.error("No users found in the database or error accessing user_profiles table")
            return
            
        # Create sidebar for inputs
        with st.sidebar:
            st.header("Query Parameters")
            
            # Add user ID selection
            user_id = st.selectbox("Select User ID", user_ids)
            
            # Add default dates - last 7 days
            default_end_date = pd.Timestamp.now().date()
            default_start_date = default_end_date - pd.Timedelta(days=7)
            
            start_date = st.date_input("Start Date", value=default_start_date)
            end_date = st.date_input("End Date", value=default_end_date)
            
            # Always use hourly granularity
            granularity = "Hour"
            
            # Validate date range
            if start_date > end_date:
                st.error("Start date must be before end date")
            else:
                # Auto-refresh option
                auto_refresh = st.checkbox("Auto-refresh", value=False)
                if auto_refresh:
                    st.write("Plot will automatically update")
                    refresh_interval = st.slider("Refresh interval (seconds)", 
                                               min_value=30, max_value=300, value=60, step=30)
                    # Use Streamlit's auto-refresh feature
                    st.write(f"Refreshing every {refresh_interval} seconds")
                    st.empty()  # This is needed for the rerun to work properly
                    
                # Add a button to trigger the query if not auto-refreshing
                if not auto_refresh:
                    query_button = st.button("Generate Plot")
                else:
                    query_button = True  # Always generate plot in auto-refresh mode
        
        # Display information about the app
        if not query_button:
            st.info("Select a user and date range, then click 'Generate Plot' to visualize sensor data.")
        
        # Main content area
        if query_button and start_date <= end_date:
            # Add a progress indicator
            progress_bar = st.progress(0)
            
            try:
                with st.spinner("Fetching data and generating plot..."):
                    # Plot sensor data
                    fig = plot_sensor_data(
                        engine=engine,
                        user_id=user_id,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    progress_bar.progress(100)
                    
                    # Display the plot with full interactivity
                    st.plotly_chart(fig, use_container_width=True, config={
                        'displayModeBar': True,
                        'scrollZoom': True,
                        'showTips': True,
                        'modeBarButtonsToAdd': ['drawline', 'selectbox'],
                        'modeBarButtonsToRemove': ['lasso2d']
                    })
                    
                    # Show summary of data
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("User ID", user_id)
                    with col2:
                        days_diff = (end_date - start_date).days + 1
                        st.metric("Time Period", f"{days_diff} days")
                    
                    # If auto-refresh is enabled, sleep and rerun
                    if 'auto_refresh' in locals() and auto_refresh:
                        import time
                        time.sleep(refresh_interval)
                        st.experimental_rerun()
                    
            except Exception as e:
                st.error(f"Error generating plot: {str(e)}")
                logger.error(f"Plot generation error: {e}", exc_info=True)
    
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        logger.error(f"Database error: {e}", exc_info=True)
    finally:
        if 'engine' in locals():
            engine.dispose()


if __name__ == "__main__":
    main()