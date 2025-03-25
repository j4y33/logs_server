# streamlit_sensor_data.py
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sqlalchemy
from sqlalchemy.pool import QueuePool
import sqlalchemy.sql.expression as sql
import streamlit as st
from matplotlib.figure import Figure
import io
import matplotlib.dates as mdates

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger()

# Set Seaborn style
sns.set_style("dark")

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
            # Convert UUIDs to strings
            return [str(row[0]) for row in result]
    except Exception as e:
        logger.error(f"Error fetching user IDs: {e}")
        return []

def plot_sensor_data(
    engine,
    user_id,
    start_date,
    end_date
):
    """
    Plot sensor data over time for a specific user and date range using matplotlib.
    
    Args:
        engine: SQLAlchemy database engine
        user_id: User ID to analyze
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        matplotlib figure
    """
    # Convert dates to strings for SQL query
    start_str = pd.Timestamp(start_date).strftime('%Y-%m-%d')
    end_str = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Sensor tables to query
    tables = {
        'accelerometer_data': 'Accelerometer',
        'gyroscope_data': 'Gyroscope',
        'magnetometer_data': 'Magnetometer',
        'locations': 'Location'
    }
    
    # Create figure with dark style
    plt.style.use('dark_background')
    fig, axes = plt.subplots(len(tables), 1, figsize=(14, 12), sharex=True)
    
    # Enhanced colors for better visibility
    colors = ['#3a86ff', '#4cc9f0', '#f72585', '#7209b7']
    bar_colors = ['#3a86ff', '#4cc9f0', '#f72585', '#7209b7']
    
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
                # Convert to datetime and create a simple count
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.info(f"Found {len(df)} records in {table}")
                
                # Create density plot of timestamps
                if len(axes) > 1:
                    ax = axes[i]
                else:
                    ax = axes  # Handle case with only one table
                
                # Don't show individual points if there are too many
                if len(df) < 5000:
                    ax.plot(df['timestamp'], [1] * len(df), 'o', 
                            markersize=2, alpha=0.6, color=colors[i])
                
                # Add hourly count as histogram with improved styling
                df['hour'] = df['timestamp'].dt.floor('h')
                hourly_count = df.groupby('hour').size()
                
                ax2 = ax.twinx()
                bars = ax2.bar(hourly_count.index, hourly_count.values, 
                        width=1/24, alpha=0.7, color=bar_colors[i], edgecolor='none')
                
                # Improved axis styling
                ax2.set_ylabel('Hourly count', color='white', fontsize=12)
                ax2.tick_params(axis='y', colors='white', labelsize=11)
                ax2.grid(False)  # Remove grid on right y-axis
                
                # Add record count to plot with better visibility
                ax.text(0.01, 0.93, f"Total: {len(df):,} records", 
                        transform=ax.transAxes, fontsize=12, color='white',
                        bbox=dict(facecolor='black', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
                
                # Set y label for the main plot
                ax.set_ylabel(label, color='white', fontsize=12, fontweight='bold')
                ax.set_ylim(0, 2)
                ax.set_yticks([])  # Hide y-ticks since they're not meaningful
                
                # Improve x-axis readability
                ax.tick_params(axis='x', colors='white', labelsize=11)
                
                # Enhanced grid for better readability
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                
                # Add background shading to bars for better contrast
                for bar in bars:
                    bar.set_zorder(0)  # Put bars behind other elements
                
            else:
                logger.warning(f"No data found for {table}")
                if len(axes) > 1:
                    axes[i].text(0.5, 0.5, 'No data available', 
                               ha='center', va='center', transform=axes[i].transAxes, 
                               color='white', fontsize=12)
                    axes[i].set_ylabel(label, color='white', fontsize=12, fontweight='bold')
                    axes[i].tick_params(axis='y', colors='white')
                else:
                    axes.text(0.5, 0.5, 'No data available', 
                           ha='center', va='center', transform=axes.transAxes, 
                           color='white', fontsize=12)
                    axes.set_ylabel(label, color='white', fontsize=12, fontweight='bold')
                    axes.tick_params(axis='y', colors='white')
                    
        except Exception as e:
            logger.error(f"Error processing {table}: {e}")
            if len(axes) > 1:
                axes[i].text(0.5, 0.5, f'Error: {str(e)}', 
                           ha='center', va='center', transform=axes[i].transAxes, 
                           color='red', fontsize=12)
            else:
                axes.text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=axes.transAxes, 
                       color='red', fontsize=12)
    
    # Configure x-axis date formatting
    if len(tables) > 1:
        ax = axes[-1]  # Get the last subplot for x-axis formatting
    else:
        ax = axes
        
    # Set better date formatting
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    # Add title and adjust layout
    fig.suptitle(f'Sensor Data for User {user_id}\n{start_date} to {end_date}', 
                 color='white', fontsize=16, y=0.98)
    plt.xlabel('Timestamp', color='white', fontsize=12, labelpad=10)
    plt.subplots_adjust(hspace=0.3, top=0.93, bottom=0.08, left=0.08, right=0.92)
    
    return fig

def main():
    st.set_page_config(
        page_title="Sensor Data Viewer", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for better styling
    st.markdown("""
        <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
        }
        .stPlotlyChart, .stButton {
            margin-bottom: 2rem;
        }
        h1 {
            margin-bottom: 1.5rem;
        }
        .stMetric {
            background-color: rgba(0, 0, 0, 0.1);
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Sensor Data Visualization Tool")
    st.write("Visualize sensor data patterns across different time periods")
    
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
            
            # User ID selection - display full ID without prefix
            user_id = st.selectbox("Select User ID", user_ids)
            
            # Add default dates - last 7 days
            default_end_date = pd.Timestamp.now().date()
            default_start_date = default_end_date - pd.Timedelta(days=7)
            
            start_date = st.date_input("Start Date", value=default_start_date)
            end_date = st.date_input("End Date", value=default_end_date)
            
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
            st.write("This application visualizes sensor data from accelerometer, gyroscope, magnetometer, and location sensors.")
        
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
                    
                    # Display the plot
                    st.pyplot(fig)
                    
                    # Show summary of data
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("User ID", user_id)
                    with col2:
                        days_diff = (end_date - start_date).days + 1
                        st.metric("Time Period", f"{days_diff} days")
                    
                    # Add download button
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label="Download Plot",
                        data=buf,
                        file_name=f"sensor_data_{user_id}_{start_date}_{end_date}.png",
                        mime="image/png"
                    )
                    
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