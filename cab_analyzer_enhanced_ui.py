import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
import math
import json
import io

# Route configuration
ROUTE_CONFIG = {
    1: {"name": "Gangtok→Bagdogra", "states": ["Sikkim", "West Bengal"]},
    2: {"name": "Bagdogra→Gangtok", "states": ["West Bengal", "Sikkim"]},
    3: {"name": "Darjeeling→Bagdogra", "states": ["West Bengal"]},
    4: {"name": "Bagdogra→Darjeeling", "states": ["West Bengal"]},
    7: {"name": "Kalimpong→Bagdogra", "states": ["West Bengal"]},
    8: {"name": "Bagdogra→Kalimpong", "states": ["West Bengal"]},
    9: {"name": "Shillong→Guwahati", "states": ["Meghalaya", "Assam"]},
    10: {"name": "Guwahati→Shillong", "states": ["Assam", "Meghalaya"]}
}

ROUTE_PAIRS = {
    "Gangtok ↔ Bagdogra (1-2)": [1, 2],
    "Darjeeling ↔ Bagdogra (3-4)": [3, 4],
    "Kalimpong ↔ Bagdogra (7-8)": [7, 8],
    "Shillong ↔ Guwahati (9-10)": [9, 10]
}

# Set page configuration
st.set_page_config(page_title="Cab Schedule Analyzer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #1e3a8a;
        font-weight: 600;
    }
    .css-1d391kg {
        background-color: #e6efff;
        padding: 20px;
    }
    .stSidebar .stSelectbox, .stSidebar .stRadio {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton button {
        background-color: #3b82f6;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stButton button:hover {
        background-color: #2563eb;
    }
    .stDataFrame {
        border: 1px solid #e5e7eb;
        border-radius: 5px;
        overflow: hidden;
    }
    .stDataFrame table {
        width: 100%;
        border-collapse: collapse;
    }
    .stDataFrame th, .stDataFrame td {
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid #e5e7eb;
    }
    .stDataFrame th {
        background-color: #f3f4f6;
        font-weight: 600;
        color: #1f2937;
    }
    .stDataFrame tr:hover {
        background-color: #f9fafb;
    }
    .stExpander {
        border: 1px solid #e5e7eb;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 16px;
        text-align: center;
        margin: 10px;
    }
    .metric-card h4 {
        margin: 0;
        font-size: 16px;
        color: #1e3a8a;
    }
    .metric-card p {
        margin: 8px 0 0;
        font-size: 24px;
        font-weight: bold;
        color: #111827;
    }
    .highlight-above-high {
        background-color: #d1fae5 !important;
    }
    .highlight-below-low {
        background-color: #fee2e2 !important;
    }
    .highlight-normal {
        background-color: #fef3c7 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Function to load and clean data
@st.cache_data
def load_and_clean_data(uploaded_file):
    try:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext == '.csv':
            df = pd.read_csv(uploaded_file)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file type. Please upload a .csv, .xlsx, or .xls file.")
        
        required_columns = ['Car Number', 'Route ID', 'Travel Date', 'Travel Time', 'Seats Booked']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        df['Travel Date'] = pd.to_datetime(df['Travel Date'], format='%d-%m-%Y', errors='coerce')
        df['Seats Booked'] = pd.to_numeric(df['Seats Booked'], errors='coerce').fillna(0).astype(int)
        df['Travel Time'] = df['Travel Time'].str.strip()
        df = df.dropna(subset=['Car Number', 'Route ID', 'Travel Date', 'Travel Time'])
        
        df['Route ID'] = df['Route ID'].astype(str)
        df['Parsed Time'] = pd.to_datetime(df['Travel Time'], errors='coerce', format='%I:%M %p')
        df['Hour'] = df['Parsed Time'].dt.hour.where(df['Parsed Time'].notna(), pd.to_datetime(df['Travel Time'], errors='coerce').dt.hour)
        df = df.dropna(subset=['Hour'])
        df['Hour'] = df['Hour'].astype(int)
        df['Formatted Time'] = df['Travel Time']
        df = df.drop(columns=['Parsed Time'])
        
        df['Route Name'] = df['Route ID'].map(lambda x: ROUTE_CONFIG.get(int(x), {}).get('name', 'Unknown'))
        route_pair_map = {str(rid): pair_name for pair_name, route_ids in ROUTE_PAIRS.items() for rid in route_ids}
        df['Route Pair'] = df['Route ID'].map(route_pair_map)
        df['Day of Week'] = df['Travel Date'].dt.day_name()
        
        st.write(f"Unique Travel Time formats: {df['Travel Time'].unique()[:10]}")
        
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Function to convert DataFrame to CSV for download
def convert_df_to_csv(df, date_columns=None, percent_columns=None):
    df_copy = df.copy()
    if date_columns:
        for col in date_columns:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].dt.strftime('%d-%m-%Y')
    if percent_columns:
        for col in percent_columns:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].apply(lambda x: f"{x:.2f}%")
    return df_copy.to_csv(index=False, encoding='utf-8')

# Analysis function
@st.cache_data
def analyze_data(df, analysis_type, selection, date_range, high_threshold, low_threshold):
    insights = []
    hour_insights = {}
    recommendations = []
    max_capacity = 5
    target_occupancy_high = 0.7
    target_occupancy_low = 0.5
    
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    if analysis_type == "Single Route":
        route_id = selection.split(':')[0].strip()
        filtered_df = df[(df['Route ID'] == route_id) & 
                        (df['Travel Date'].dt.date >= start_date.date()) & 
                        (df['Travel Date'].dt.date <= end_date.date())]
        selection_name = ROUTE_CONFIG[int(route_id)]['name']
        route_ids = [route_id]
    else:
        route_ids = [str(rid) for rid in ROUTE_PAIRS[selection]]
        filtered_df = df[(df['Route ID'].isin(route_ids)) & 
                        (df['Travel Date'].dt.date >= start_date.date()) & 
                        (df['Travel Date'].dt.date <= end_date.date())]
        selection_name = selection
    
    if filtered_df.empty:
        st.warning(f"No data available for {selection_name} in the selected date range.")
        return None, [], selection_name, None, None, None, None, None, {}, []
    
    for rid in filtered_df['Route ID'].unique():
        st.write(f"Rows for Route ID {rid} ({ROUTE_CONFIG.get(int(rid), {}).get('name', 'Unknown')}): {len(filtered_df[filtered_df['Route ID'] == rid])}")
    
    total_days = filtered_df['Travel Date'].dt.date.nunique()
    total_seats_booked = filtered_df['Seats Booked'].sum()
    total_seats_available = len(filtered_df) * max_capacity
    occupancy_percentage = (total_seats_booked / total_seats_available * 100) if total_seats_available > 0 else 0
    summary_data = pd.DataFrame({
        'Total Number of Days': [total_days],
        'Total Seats Booked': [total_seats_booked],
        'Total Seats Available': [total_seats_available],
        'Percentage Occupancy': [round(occupancy_percentage, 2)]
    })
    insights.append(f"Overall Occupancy for {selection_name}: {occupancy_percentage:.2f}%")
    
    day_summary = filtered_df.groupby('Day of Week').agg({
        'Seats Booked': 'sum',
        'Travel Date': 'size'
    }).reset_index()
    day_summary.columns = ['Day of Week', 'Seats Booked', 'Trip Count']
    day_summary['Total Capacity'] = day_summary['Trip Count'] * max_capacity
    day_summary['Occupancy (%)'] = (day_summary['Seats Booked'] / day_summary['Total Capacity'] * 100).round(2)
    day_summary['Highlight'] = day_summary['Occupancy (%)'].apply(
        lambda x: 'Above High' if x > high_threshold else 'Below Low' if x < low_threshold else 'Normal'
    )
    day_summary = day_summary.sort_values('Occupancy (%)', ascending=False)
    if not day_summary.empty:
        highest_day = day_summary.iloc[0]
        lowest_day = day_summary.iloc[-1]
        insights.append(f"Highest Booking Day: {highest_day['Day of Week']} ({highest_day['Occupancy (%)']:.2f}% occupancy)")
        insights.append(f"Lowest Booking Day: {lowest_day['Day of Week']} ({lowest_day['Occupancy (%)']:.2f}% occupancy)")
        insights.append(f"Scheduling Recommendation: Increase car availability on {highest_day['Day of Week']} "
                        f"due to high occupancy ({highest_day['Occupancy (%)']:.2f}%).")
        insights.append(f"Scheduling Recommendation: Reduce car availability on {lowest_day['Day of Week']} "
                        f"due to low occupancy ({lowest_day['Occupancy (%)']:.2f}%).")
    
    date_occupancy = filtered_df.groupby('Travel Date').agg({
        'Seats Booked': 'sum',
        'Day of Week': 'first',
        'Car Number': 'count'
    }).reset_index()
    date_occupancy.columns = ['Travel Date', 'Seats Booked', 'Day of Week', 'Trip Count']
    date_occupancy['Occupancy (%)'] = (date_occupancy['Seats Booked'] / (date_occupancy['Trip Count'] * max_capacity) * 100).round(2)
    date_occupancy['Highlight'] = date_occupancy['Occupancy (%)'].apply(
        lambda x: 'Above High' if x > high_threshold else 'Below Low' if x < low_threshold else 'Normal'
    )
    
    car_summary = filtered_df.groupby('Car Number').agg({
        'Seats Booked': 'sum',
        'Travel Date': ['count', 'nunique', lambda x: list(x.dt.date)],
    }).reset_index()
    car_summary.columns = ['Car Number', 'Seats Booked', 'Trip Count', 'Days Operational Temp', 'Travel Dates']
    
    all_dates = [start_date + timedelta(days=x) for x in range((end_date.date() - start_date.date()).days + 1)]
    
    if analysis_type == "Route Pair":
        car_summary['Days Operational'] = 0.0
        car_summary['Half-Day Service Dates'] = ''
        
        for idx, row in car_summary.iterrows():
            car = row['Car Number']
            car_dates = filtered_df[filtered_df['Car Number'] == car][['Travel Date', 'Route ID']]
            car_dates['Travel Date'] = car_dates['Travel Date'].dt.date
            days_operational = 0.0
            half_day_dates = []
            
            for date in all_dates:
                date_routes = car_dates[car_dates['Travel Date'] == date.date()]['Route ID'].unique()
                if len(date_routes) == 1 and date_routes[0] in route_ids:
                    days_operational += 0.5
                    half_day_dates.append(date.strftime('%d-%m-%Y'))
                elif len(date_routes) == 2 and all(rid in route_ids for rid in date_routes):
                    days_operational += 1.0
            
            car_summary.at[idx, 'Days Operational'] = days_operational
            car_summary.at[idx, 'Half-Day Service Dates'] = ', '.join(half_day_dates[:5]) + ('...' if len(half_day_dates) > 5 else '')
    else:
        car_summary['Days Operational'] = car_summary['Days Operational Temp']
        car_summary['Half-Day Service Dates'] = ''
    
    total_date_range_days = (end_date.date() - start_date.date()).days + 1
    car_summary['Days Not in Service'] = total_date_range_days - car_summary['Days Operational']
    
    car_non_service_dates = {}
    for car in car_summary['Car Number']:
        car_dates = filtered_df[filtered_df['Car Number'] == car]['Travel Date'].dt.date.unique()
        non_service_dates = [d.strftime('%d-%m-%Y') for d in all_dates if d.date() not in car_dates]
        car_non_service_dates[car] = ', '.join(non_service_dates[:5]) + ('...' if len(non_service_dates) > 5 else '')
    car_summary['Non-Service Dates'] = car_summary['Car Number'].map(car_non_service_dates)
    
    car_summary['Total Seats Available'] = car_summary['Trip Count'] * max_capacity
    car_summary['Percentage Occupancy'] = (car_summary['Seats Booked'] / car_summary['Total Seats Available'] * 100).round(2)
    car_summary['Highlight'] = car_summary['Percentage Occupancy'].apply(
        lambda x: 'Above High' if x > high_threshold else 'Below Low' if x < low_threshold else 'Normal'
    )
    car_summary = car_summary.sort_values('Percentage Occupancy', ascending=False)
    
    if not car_summary.empty:
        highest_car = car_summary.iloc[0]
        lowest_car = car_summary.iloc[-1]
        insights.append(f"Highest Utilized Car: {highest_car['Car Number']} ({highest_car['Percentage Occupancy']:.2f}% occupancy)")
        insights.append(f"Lowest Utilized Car: {lowest_car['Car Number']} ({lowest_car['Percentage Occupancy']:.2f}% occupancy)")
        insights.append(f"Scheduling Recommendation: Prioritize maintenance or reassign {lowest_car['Car Number']} "
                        f"due to low utilization ({lowest_car['Percentage Occupancy']:.2f}%).")
    
    # Booking Trend by Hour (Overall, no Day of Week)
    hour_summary_list = []
    for rid in route_ids:
        route_df = filtered_df[filtered_df['Route ID'] == rid]
        hour_summary = route_df.groupby('Travel Time').agg({
            'Seats Booked': 'sum',
            'Car Number': 'count'
        }).reset_index()
        hour_summary.columns = ['Time', 'Total Seats Booked', 'Number of Trips']
        hour_summary['Total Seats Available'] = hour_summary['Number of Trips'] * max_capacity
        hour_summary['Percentage Occupancy'] = (hour_summary['Total Seats Booked'] / hour_summary['Total Seats Available'] * 100).fillna(0).round(2)
        hour_summary['Highlight'] = hour_summary['Percentage Occupancy'].apply(
            lambda x: 'Above High' if x > high_threshold else 'Below Low' if x < low_threshold else 'Normal'
        )
        hour_summary['Route Name'] = ROUTE_CONFIG[int(rid)]['name']
        hour_summary = hour_summary.sort_values('Time')
        hour_summary_list.append(hour_summary)
        
        # Generate hourly insights
        if not hour_summary.empty:
            route_name = ROUTE_CONFIG[int(rid)]['name']
            max_hour = hour_summary.loc[hour_summary['Percentage Occupancy'].idxmax()]
            min_hour = hour_summary.loc[hour_summary['Percentage Occupancy'].idxmin()]
            high_demand = hour_summary[hour_summary['Percentage Occupancy'] > high_threshold]['Time'].values.tolist()
            low_demand = hour_summary[hour_summary['Percentage Occupancy'] < low_threshold]['Time'].values.tolist()
            total_trips = hour_summary['Number of Trips'].sum()
            top_hours = hour_summary.nlargest(3, 'Number of Trips')['Time'].values.tolist()
            trip_concentration = hour_summary.nlargest(3, 'Number of Trips')['Number of Trips'].sum() / total_trips * 100 if total_trips > 0 else 0
            
            hour_insights[route_name] = []
            hour_insights[route_name].append(f"Peak Time: {max_hour['Time']} ({max_hour['Percentage Occupancy']:.2f}% occupancy)")
            hour_insights[route_name].append(f"Low-Performing Time: {min_hour['Time']} ({min_hour['Percentage Occupancy']:.2f}% occupancy)")
            if high_demand:
                hour_insights[route_name].append(f"High-Demand Times: {', '.join(high_demand)} (above {high_threshold}% occupancy, consider adding trips)")
            if low_demand:
                hour_insights[route_name].append(f"Low-Demand Times: {', '.join(low_demand)} (below {low_threshold}% occupancy, consider reducing trips)")
            if top_hours and total_trips > 0:
                hour_insights[route_name].append(f"Trip Distribution: {trip_concentration:.1f}% of trips occur at times {', '.join(top_hours)}")
    
    hour_summary = pd.concat(hour_summary_list, ignore_index=True) if hour_summary_list else pd.DataFrame()
    
    # Scheduling Recommendations (Overall, car assignments)
    min_trips_for_recommendation = 3
    available_cars = car_summary[['Car Number', 'Percentage Occupancy']].sort_values('Percentage Occupancy')
    num_unique_cars = len(available_cars)
    
    for rid in route_ids:
        route_name = ROUTE_CONFIG[int(rid)]['name']
        route_hour_summary = hour_summary[hour_summary['Route Name'] == route_name]
        for _, row in route_hour_summary.iterrows():
            if row['Number of Trips'] < min_trips_for_recommendation:
                continue
            current_trips = row['Number of Trips']
            seats_booked = row['Total Seats Booked']
            current_occupancy = row['Percentage Occupancy']
            time = row['Time']
            
            if current_occupancy > high_threshold:
                trips_to_add = math.ceil((seats_booked - target_occupancy_high * current_trips * max_capacity) / max_capacity)
                if trips_to_add > 0:
                    suggested_cars = available_cars['Car Number'].head(trips_to_add).tolist()
                    if suggested_cars:
                        recommendations.append({
                            'Time': time,
                            'Route': route_name,
                            'Current Trips': current_trips,
                            'Current Occupancy': current_occupancy,
                            'Action': 'Add',
                            'Suggested Trips': trips_to_add,
                            'Suggested Cars': ', '.join(suggested_cars) if suggested_cars else 'None'
                        })
                        hour_insights[route_name].append(
                            f"Scheduling Recommendation: Add {trips_to_add} trip(s) at {time} "
                            f"(current {current_occupancy:.2f}%) using cars {', '.join(suggested_cars)}"
                        )
            elif current_occupancy < low_threshold and current_trips > 1:
                trips_to_remove = math.floor((current_trips * max_capacity * current_occupancy - seats_booked) / max_capacity)
                if trips_to_remove > 0:
                    recommendations.append({
                        'Time': time,
                        'Route': route_name,
                        'Current Trips': current_trips,
                        'Current Occupancy': current_occupancy,
                        'Action': 'Remove',
                        'Suggested Trips': trips_to_remove,
                        'Suggested Cars': 'N/A'
                    })
                    hour_insights[route_name].append(
                        f"Scheduling Recommendation: Remove {trips_to_remove} trip(s) at {time} "
                        f"(current {current_occupancy:.2f}%)"
                    )
    
    # Car Scheduling Recommendations
    for rid in route_ids:
        route_name = ROUTE_CONFIG[int(rid)]['name']
        route_hour_summary = hour_summary[hour_summary['Route Name'] == route_name]
        high_demand = route_hour_summary[route_hour_summary['Percentage Occupancy'] > high_threshold][['Time', 'Percentage Occupancy', 'Number of Trips']].sort_values('Percentage Occupancy', ascending=False)
        
        # Assign cars to top time slots
        top_times = high_demand.head(num_unique_cars)[['Time', 'Percentage Occupancy']].values.tolist()
        car_assignments = []
        assigned_cars = available_cars['Car Number'].head(len(top_times)).tolist()
        
        for i, (time, occupancy) in enumerate(top_times):
            if i < len(assigned_cars):
                car = assigned_cars[i]
                recommendations.append({
                    'Time': time,
                    'Route': route_name,
                    'Current Trips': route_hour_summary[route_hour_summary['Time'] == time]['Number of Trips'].iloc[0],
                    'Current Occupancy': occupancy,
                    'Action': 'Schedule',
                    'Suggested Trips': 1,
                    'Suggested Cars': car
                })
                hour_insights[route_name].append(
                    f"Scheduling Recommendation: Schedule car {car} at {time} "
                    f"(current {occupancy:.2f}% occupancy)"
                )
            else:
                break
        
        # Note unassigned cars
        unassigned_cars = available_cars['Car Number'].iloc[len(top_times):].tolist()
        if unassigned_cars:
            hour_insights[route_name].append(
                f"Scheduling Recommendation: {len(unassigned_cars)} car(s) ({', '.join(unassigned_cars)}) not assigned due to insufficient high-demand time slots"
            )
    
    recommendations_df = pd.DataFrame(recommendations) if recommendations else pd.DataFrame(columns=['Time', 'Route', 'Current Trips', 'Current Occupancy', 'Action', 'Suggested Trips', 'Suggested Cars'])
    
    return filtered_df, insights, selection_name, date_occupancy, hour_summary, summary_data, day_summary, car_summary, hour_insights, recommendations_df

# Main Streamlit app
def main():
    st.title("🚗 Cab Schedule Analytical Dashboard")
    st.markdown("Upload a .csv, .xlsx, or .xls file to analyze cab schedules for a single route or route pair.")
    
    with st.sidebar:
        st.header("Analysis Options")
        uploaded_file = st.file_uploader("📂 Upload Data File", type=['csv', 'xlsx', 'xls'], help="Upload a .csv, .xlsx, or .xls file containing cab schedule data.")
        
        if uploaded_file:
            analysis_type = st.radio("Analysis Type", ["Single Route", "Route Pair"], help="Choose to analyze a single route or a pair of routes.")
            if analysis_type == "Single Route":
                route_options = [f"{rid}: {details['name']}" for rid, details in ROUTE_CONFIG.items() if rid in [1, 2, 3, 4, 7, 8, 9, 10]]
                selection = st.selectbox("Select Route", options=route_options, help="Select a single route for analysis.")
            else:
                selection = st.selectbox("Select Route Pair", options=list(ROUTE_PAIRS.keys()), help="Select a route pair for analysis.")
            
            high_threshold = st.slider("High Occupancy Threshold (%)", 50.0, 100.0, 75.0, help="Set the threshold for high-demand periods")
            low_threshold = st.slider("Low Occupancy Threshold (%)", 0.0, 50.0, 50.0, help="Set the threshold for low-demand periods")
            
            df = load_and_clean_data(uploaded_file)
            if df is not None:
                date_range = st.date_input("📅 Select Date Range", 
                                           [df['Travel Date'].min(), df['Travel Date'].max()],
                                           help="Choose the date range for analysis.")
                
                if st.button("🗑️ Reset", help="Clear file and reset filters"):
                    st.rerun()
    
    if uploaded_file is None:
        st.info("Please upload a file to begin analysis.")
        return
    
    if df is None:
        return
    
    if len(date_range) != 2:
        st.warning("Please select a valid date range.")
        return
    
    with st.container():
        filtered_df, insights, selection_name, date_occupancy, hour_summary, summary_data, day_summary, car_summary, hour_insights, recommendations_df = analyze_data(
            df, analysis_type, selection, date_range, high_threshold, low_threshold)
        
        if filtered_df is None or filtered_df.empty:
            st.warning(f"No data available for {selection_name} in the selected date range.")
            return
        
        col1, col2 = st.columns([1, 1])
        with col1:
            with st.expander("📊 Overall Data Summary", expanded=True):
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h4>📅 Total Number of Days</h4>
                            <p>{summary_data['Total Number of Days'][0]}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class="metric-card">
                            <h4>🪑 Total Seats Booked</h4>
                            <p>{summary_data['Total Seats Booked'][0]}</p>
                        </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h4>🚗 Total Seats Available</h4>
                            <p>{summary_data['Total Seats Available'][0]}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class="metric-card">
                            <h4>📈 Percentage Occupancy</h4>
                            <p>{summary_data['Percentage Occupancy'][0]:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            with st.expander("💡 Key Insights", expanded=True):
                for insight in insights:
                    st.markdown(f"- {insight}")
        
        st.divider()
        
        with st.expander("🔍 Data Overview"):
            st.dataframe(filtered_df[['Car Number', 'Route ID', 'Route Name', 'Travel Date', 'Travel Time', 'Seats Booked', 'Day of Week']],
                         use_container_width=True)
            if not filtered_df.empty:
                csv = convert_df_to_csv(filtered_df[['Car Number', 'Route ID', 'Route Name', 'Travel Date', 'Travel Time', 'Seats Booked', 'Day of Week']],
                                        date_columns=['Travel Date'])
                st.download_button(
                    label="📥 Download Data Overview as CSV",
                    data=csv,
                    file_name=f"data_overview_{selection_name.replace(' ↔ ', '_').lower()}.csv",
                    mime="text/csv",
                    help="Download the Data Overview table as a CSV file"
                )
        
        st.divider()
        
        st.header("🚙 Car Utilization Analysis")
        if car_summary is not None and not car_summary.empty:
            def highlight_row(row):
                if row['Highlight'] == 'Above High':
                    return ['background-color: #d1fae5'] * len(row)
                elif row['Highlight'] == 'Below Low':
                    return ['background-color: #fee2e2'] * len(row)
                else:
                    return ['background-color: #fef3c7'] * len(row)
            
            styled_car_summary = car_summary[['Car Number', 'Days Operational', 'Trip Count', 'Days Not in Service', 'Non-Service Dates', 'Half-Day Service Dates', 'Percentage Occupancy', 'Highlight']].style.apply(highlight_row, axis=1)
            st.dataframe(styled_car_summary,
                         column_config={
                             "Days Operational": st.column_config.NumberColumn(format="%.1f"),
                             "Days Not in Service": st.column_config.NumberColumn(format="%.1f"),
                             "Percentage Occupancy": st.column_config.NumberColumn(format="%.2f%%")
                         },
                         use_container_width=True)
            csv = convert_df_to_csv(car_summary[['Car Number', 'Days Operational', 'Trip Count', 'Days Not in Service', 'Non-Service Dates', 'Half-Day Service Dates', 'Percentage Occupancy']],
                                    percent_columns=['Percentage Occupancy'])
            st.download_button(
                label="📥 Download Car Utilization as CSV",
                data=csv,
                file_name=f"car_utilization_{selection_name.replace(' ↔ ', '_').lower()}.csv",
                mime="text/csv",
                help="Download the Car Utilization Analysis table as a CSV file"
            )
        else:
            st.warning("No data available for Car Utilization Analysis.")
        
        st.divider()
        
        st.header("📅 Booking Trend by Day of Week")
        if day_summary is not None and not day_summary.empty:
            def highlight_row(row):
                if row['Highlight'] == 'Above High':
                    return ['background-color: #d1fae5'] * len(row)
                elif row['Highlight'] == 'Below Low':
                    return ['background-color: #fee2e2'] * len(row)
                else:
                    return ['background-color: #fef3c7'] * len(row)
            
            styled_day_summary = day_summary[['Day of Week', 'Seats Booked', 'Trip Count', 'Total Capacity', 'Occupancy (%)', 'Highlight']].style.apply(highlight_row, axis=1)
            st.dataframe(styled_day_summary,
                         column_config={
                             "Occupancy (%)": st.column_config.NumberColumn(format="%.2f%%")
                         },
                         use_container_width=True)
            csv = convert_df_to_csv(day_summary[['Day of Week', 'Seats Booked', 'Trip Count', 'Total Capacity', 'Occupancy (%)']],
                                    percent_columns=['Occupancy (%)'])
            st.download_button(
                label="📥 Download Booking Trend by Day as CSV",
                data=csv,
                file_name=f"booking_trend_by_day_{selection_name.replace(' ↔ ', '_').lower()}.csv",
                mime="text/csv",
                help="Download the Booking Trend by Day of Week table as a CSV file"
            )
        else:
            st.warning("No data available for Booking Trend by Day of Week.")
        
        st.divider()
        
        st.header("📆 Percentage Occupancy by Date")
        if date_occupancy is not None and not date_occupancy.empty:
            def highlight_row(row):
                if row['Highlight'] == 'Above High':
                    return ['background-color: #d1fae5'] * len(row)
                elif row['Highlight'] == 'Below Low':
                    return ['background-color: #fee2e2'] * len(row)
                else:
                    return ['background-color: #fef3c7'] * len(row)
            
            styled_df = date_occupancy[['Travel Date', 'Day of Week', 'Seats Booked', 'Trip Count', 'Occupancy (%)', 'Highlight']].style.apply(highlight_row, axis=1)
            st.dataframe(styled_df,
                         column_config={
                             "Travel Date": st.column_config.DateColumn(format="DD-MM-YYYY"),
                             "Occupancy (%)": st.column_config.NumberColumn(format="%.2f%%")
                         },
                         use_container_width=True)
            csv = convert_df_to_csv(date_occupancy[['Travel Date', 'Day of Week', 'Seats Booked', 'Trip Count', 'Occupancy (%)']],
                                    date_columns=['Travel Date'], percent_columns=['Occupancy (%)'])
            st.download_button(
                label="📥 Download Occupancy by Date as CSV",
                data=csv,
                file_name=f"occupancy_by_date_{selection_name.replace(' ↔ ', '_').lower()}.csv",
                mime="text/csv",
                help="Download the Percentage Occupancy by Date table as a CSV file"
            )
        else:
            st.warning("No data available for Percentage Occupancy by Date.")
        
        st.divider()
        
        st.header("⏰ Booking Trend by Hour")
        if analysis_type == "Single Route":
            if hour_summary is not None and not hour_summary.empty:
                st.subheader(f"Booking Trend by Hour - {selection_name}")
                def highlight_row(row):
                    if row['Highlight'] == 'Above High':
                        return ['background-color: #d1fae5'] * len(row)
                    elif row['Highlight'] == 'Below Low':
                        return ['background-color: #fee2e2'] * len(row)
                    else:
                        return ['background-color: #fef3c7'] * len(row)
                
                styled_hour_summary = hour_summary[['Time', 'Number of Trips', 'Total Seats Available', 'Total Seats Booked', 'Percentage Occupancy', 'Highlight']].style.apply(highlight_row, axis=1)
                st.dataframe(styled_hour_summary,
                             column_config={
                                 "Time": st.column_config.TextColumn("Time"),
                                 "Percentage Occupancy": st.column_config.NumberColumn(format="%.2f%%")
                             },
                             use_container_width=True)
                csv = convert_df_to_csv(hour_summary[['Time', 'Number of Trips', 'Total Seats Available', 'Total Seats Booked', 'Percentage Occupancy']],
                                        percent_columns=['Percentage Occupancy'])
                st.download_button(
                    label=f"📥 Download Booking Trend by Hour as CSV",
                    data=csv,
                    file_name=f"booking_trend_by_hour_{selection_name.replace('→', '_').lower()}.csv",
                    mime="text/csv",
                    help=f"Download the Booking Trend by Hour table for {selection_name} as a CSV file"
                )
                
                chart_data = hour_summary.groupby('Time').agg({
                    'Percentage Occupancy': 'mean',
                    'Number of Trips': 'sum'
                }).reset_index()
                chart_config = {
                    "type": "bar",
                    "data": {
                        "labels": chart_data['Time'].tolist(),
                        "datasets": [
                            {
                                "label": "Percentage Occupancy",
                                "data": chart_data['Percentage Occupancy'].tolist(),
                                "backgroundColor": ["#d1fae5" if x > high_threshold else "#fee2e2" if x < low_threshold else "#fef3c7" for x in chart_data['Percentage Occupancy']],
                                "yAxisID": "y"
                            },
                            {
                                "label": "Number of Trips",
                                "data": chart_data['Number of Trips'].tolist(),
                                "type": "line",
                                "borderColor": "#3b82f6",
                                "fill": False,
                                "yAxisID": "y1"
                            }
                        ]
                    },
                    "options": {
                        "scales": {
                            "y": {"position": "left", "title": {"display": True, "text": "Percentage Occupancy (%)"}},
                            "y1": {"position": "right", "title": {"display": True, "text": "Number of Trips"}}
                        },
                        "plugins": {"title": {"display": True, "text": f"Hourly Trends - {selection_name}"}}
                    }
                }
                st.components.v1.html(f"""
                    <div style='height:400px;'>
                        <canvas id='hourlyChart'></canvas>
                        <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
                        <script>
                            const ctx = document.getElementById('hourlyChart').getContext('2d');
                            new Chart(ctx, {json.dumps(chart_config)});
                        </script>
                    </div>
                """, height=400)
                
                with st.expander("⏰ Hourly Insights", expanded=True):
                    for insight in hour_insights.get(selection_name, []):
                        st.markdown(f"- {insight}")
            else:
                st.warning(f"No hourly data available for {selection_name}.")
        else:
            route_ids = [str(rid) for rid in ROUTE_PAIRS[selection]]
            for rid in route_ids:
                route_hour_summary = hour_summary[hour_summary['Route Name'] == ROUTE_CONFIG[int(rid)]['name']][['Time', 'Number of Trips', 'Total Seats Available', 'Total Seats Booked', 'Percentage Occupancy', 'Highlight']]
                if not route_hour_summary.empty:
                    st.subheader(f"Booking Trend by Hour - {ROUTE_CONFIG[int(rid)]['name']}")
                    def highlight_row(row):
                        if row['Highlight'] == 'Above High':
                            return ['background-color: #d1fae5'] * len(row)
                        elif row['Highlight'] == 'Below Low':
                            return ['background-color: #fee2e2'] * len(row)
                        else:
                            return ['background-color: #fef3c7'] * len(row)
                    
                    styled_hour_summary = route_hour_summary.style.apply(highlight_row, axis=1)
                    st.dataframe(styled_hour_summary,
                                 column_config={
                                     "Time": st.column_config.TextColumn("Time"),
                                     "Percentage Occupancy": st.column_config.NumberColumn(format="%.2f%%")
                                 },
                                 use_container_width=True)
                    csv = convert_df_to_csv(route_hour_summary[['Time', 'Number of Trips', 'Total Seats Available', 'Total Seats Booked', 'Percentage Occupancy']],
                                            percent_columns=['Percentage Occupancy'])
                    st.download_button(
                        label=f"📥 Download Booking Trend by Hour as CSV",
                        data=csv,
                        file_name=f"booking_trend_by_hour_{ROUTE_CONFIG[int(rid)]['name'].replace('→', '_').lower()}.csv",
                        mime="text/csv",
                        help=f"Download the Booking Trend by Hour table for {ROUTE_CONFIG[int(rid)]['name']} as a CSV file"
                    )
                    
                    chart_data = route_hour_summary.groupby('Time').agg({
                        'Percentage Occupancy': 'mean',
                        'Number of Trips': 'sum'
                    }).reset_index()
                    chart_config = {
                        "type": "bar",
                        "data": {
                            "labels": chart_data['Time'].tolist(),
                            "datasets": [
                                {
                                    "label": "Percentage Occupancy",
                                    "data": chart_data['Percentage Occupancy'].tolist(),
                                    "backgroundColor": ["#d1fae5" if x > high_threshold else "#fee2e2" if x < low_threshold else "#fef3c7" for x in chart_data['Percentage Occupancy']],
                                    "yAxisID": "y"
                                },
                                {
                                    "label": "Number of Trips",
                                    "data": chart_data['Number of Trips'].tolist(),
                                    "type": "line",
                                    "borderColor": "#3b82f6",
                                    "fill": False,
                                    "yAxisID": "y1"
                                }
                            ]
                        },
                        "options": {
                            "scales": {
                                "y": {"position": "left", "title": {"display": True, "text": "Percentage Occupancy (%)"}},
                                "y1": {"position": "right", "title": {"display": True, "text": "Number of Trips"}}
                            },
                            "plugins": {"title": {"display": True, "text": f"Hourly Trends - {ROUTE_CONFIG[int(rid)]['name']}"}}
                        }
                    }
                    st.components.v1.html(f"""
                        <div style='height:400px;'>
                            <canvas id='hourlyChart_{rid}'></canvas>
                            <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
                            <script>
                                const ctx = document.getElementById('hourlyChart_{rid}').getContext('2d');
                                new Chart(ctx, {json.dumps(chart_config)});
                            </script>
                        </div>
                    """, height=400)
                    
                    with st.expander(f"⏰ Hourly Insights - {ROUTE_CONFIG[int(rid)]['name']}", expanded=True):
                        for insight in hour_insights.get(ROUTE_CONFIG[int(rid)]['name'], []):
                            st.markdown(f"- {insight}")
                else:
                    st.warning(f"No hourly data available for {ROUTE_CONFIG[int(rid)]['name']}.")
        
        st.divider()
        
        st.header("📋 Scheduling Recommendations")
        if not recommendations_df.empty:
            def highlight_recommendation(row):
                if row['Action'] == 'Add':
                    return ['background-color: #d1fae5'] * len(row)
                elif row['Action'] == 'Remove':
                    return ['background-color: #fee2e2'] * len(row)
                else:
                    return ['background-color: #fef3c7'] * len(row)
            
            styled_recommendations = recommendations_df[['Time', 'Route', 'Current Trips', 'Current Occupancy', 'Action', 'Suggested Trips', 'Suggested Cars']].style.apply(highlight_recommendation, axis=1)
            st.dataframe(styled_recommendations,
                         column_config={
                             "Current Occupancy": st.column_config.NumberColumn(format="%.2f%%"),
                             "Suggested Trips": st.column_config.NumberColumn(format="%d")
                         },
                         use_container_width=True)
            csv = convert_df_to_csv(recommendations_df[['Time', 'Route', 'Current Trips', 'Current Occupancy', 'Action', 'Suggested Trips', 'Suggested Cars']],
                                    percent_columns=['Current Occupancy'])
            st.download_button(
                label="📥 Download Scheduling Recommendations as CSV",
                data=csv,
                file_name=f"scheduling_recommendations_{selection_name.replace(' ↔ ', '_').lower()}.csv",
                mime="text/csv",
                help="Download the Scheduling Recommendations table as a CSV file"
            )
            
            with st.expander("📋 Recommendation Insights", expanded=True):
                for route_name in hour_insights:
                    for insight in hour_insights[route_name]:
                        if "Scheduling Recommendation" in insight:
                            st.markdown(f"- {insight}")
        else:
            st.warning("No scheduling recommendations available based on current data and thresholds.")
    
    st.success("Analysis complete!")

if __name__ == "__main__":
    main()