import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import folium

from streamlit_folium import st_folium
# Load Dataset

df = pd.read_csv("zomato.csv")

# Create Cuisine List

cuisine_series = (
    df['Cuisines']
    .dropna()
    .str.split(',')
    .explode()
    .str.strip()
)

# Load Models

rating_model = joblib.load("rating_model.pkl")

scaler = joblib.load("scaler.pkl")

# Page Config

st.set_page_config(
    page_title="Restaurant Intelligence System",
    layout="wide"
)

# Sidebar Navigation

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Restaurant Recommendation",
        "Rating Prediction",
        "Cuisine Explorer",
        "Location Analysis"
    ]
)

# Dashboard

if page == "Dashboard":

    st.title("Restaurant Intelligence Dashboard")

    total_restaurants = len(df)

    avg_rating = round(
        df['Aggregate rating'].mean(),
        2
    )

    avg_cost = round(
        df['Average Cost for two'].mean(),
        2
    )

    total_cities = df['City'].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Restaurants",
        total_restaurants
    )

    col2.metric(
        "Average Rating",
        avg_rating
    )

    col3.metric(
        "Average Cost",
        avg_cost
    )

    col4.metric(
        "Cities",
        total_cities
    )

    st.divider()

    # Rating Distribution

    st.subheader("Rating Distribution")

    fig1 = px.histogram(
        df,
        x="Aggregate rating",
        nbins=20
    )

    st.plotly_chart(
        fig1,
        width='stretch'
    )

    # Top Cities

    st.subheader("Top Cities")

    top_cities = df['City'].value_counts().head(10)

    fig2 = px.bar(
        x=top_cities.index,
        y=top_cities.values,
        labels={
            'x': 'City',
            'y': 'Restaurant Count'
        }
    )

    st.plotly_chart(
        fig2,
        width='stretch'
    )

    # Top Cuisines

    st.subheader("Top Cuisines")

    top_cuisines = cuisine_series.value_counts().head(10)

    fig3 = px.bar(
        x=top_cuisines.index,
        y=top_cuisines.values,
        labels={
            'x': 'Cuisine',
            'y': 'Count'
        }
    )

    st.plotly_chart(
        fig3,
        width='stretch'
    )

    # Correlation Heatmap

    st.subheader("Correlation Heatmap")

    numerical_df = df.select_dtypes(
        include=['int64', 'float64']
    )

    corr_matrix = numerical_df.corr()

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        ax=ax
    )

    st.pyplot(fig)

# Restaurant Recommendation

elif page == "Restaurant Recommendation":

    st.title("Restaurant Recommendation System")

    # City Dropdown

    city = st.selectbox(
        "Select City",
        sorted(df['City'].dropna().unique())
    )

    # Available Cuisines Based on City

    available_cuisines = (
        df[df['City'] == city]['Cuisines']
        .dropna()
        .str.split(',')
        .explode()
        .str.strip()
        .unique()
    )

    available_cuisines = sorted(available_cuisines)

    # Cuisine Dropdown

    cuisine = st.selectbox(
        "Select Cuisine",
        available_cuisines
    )

    # Price Range

    price_range = st.slider(
        "Select Price Range",
        1,
        4,
        2
    )

    # Filter Restaurants

    filtered_df = df[
        (df['City'] == city) &
        (
            df['Cuisines']
            .str.contains(
                cuisine,
                case=False,
                na=False
            )
        )
    ]

    # Flexible Price Filtering

    filtered_df = filtered_df[
        filtered_df['Price range'].between(
            max(1, price_range - 1),
            min(4, price_range + 1)
        )
    ]

    # Sort Results

    filtered_df = filtered_df.sort_values(
        by=[
            'Aggregate rating',
            'Votes'
        ],
        ascending=False
    )

    st.subheader("Recommended Restaurants")

    # Empty Result Handling

    if filtered_df.empty:

        st.warning(
            "No restaurants found for selected filters."
        )

    else:

        st.dataframe(
            filtered_df[[
                'Restaurant Name',
                'City',
                'Cuisines',
                'Aggregate rating',
                'Average Cost for two',
                'Votes'
            ]].head(20),
            width='stretch'
        )

# Rating Prediction

elif page == "Rating Prediction":

    st.title("Restaurant Rating Prediction")

    votes = st.number_input(
        "Votes",
        min_value=0,
        value=100
    )

    avg_cost = st.number_input(
        "Average Cost for Two",
        min_value=0,
        value=500
    )

    price_range = st.slider(
        "Price Range",
        1,
        4,
        2
    )

    table_booking = st.selectbox(
        "Table Booking",
        [0, 1]
    )

    online_delivery = st.selectbox(
        "Online Delivery",
        [0, 1]
    )

    name_length = st.slider(
        "Restaurant Name Length",
        1,
        50,
        15
    )

    address_length = st.slider(
        "Address Length",
        1,
        150,
        50
    )

    input_data = pd.DataFrame({
        'Votes': [votes],
        'Average Cost for two': [avg_cost],
        'Price range': [price_range],
        'Has Table Booking': [table_booking],
        'Has Online Delivery': [online_delivery],
        'Restaurant Name Length': [name_length],
        'Address Length': [address_length]
    })

    prediction = rating_model.predict(
        input_data
    )[0]

    st.subheader("Predicted Rating")

    st.metric(
        "Aggregate Rating",
        round(prediction, 2)
    )

# Cuisine Explorer

elif page == "Cuisine Explorer":

    st.title("Cuisine Explorer")

    # Cuisine Selection

    selected_cuisine = st.selectbox(
        "Select Cuisine",
        sorted(cuisine_series.unique())
    )

    # Filter Cities Based on Selected Cuisine

    available_cities = (
        df[
            df['Cuisines']
            .str.contains(
                selected_cuisine,
                case=False,
                na=False
            )
        ]['City']
        .dropna()
        .unique()
    )

    available_cities = sorted(available_cities)

    # City Selection

    selected_city = st.selectbox(
        "Select Location",
        available_cities
    )

    # Price Range Filter

    selected_price = st.slider(
        "Select Price Range",
        1,
        4,
        2
    )

    # Filter Restaurants

    cuisine_df = df[
        (
            df['Cuisines']
            .str.contains(
                selected_cuisine,
                case=False,
                na=False
            )
        ) &
        (
            df['City'] == selected_city
        )
    ]

    # Flexible Price Filtering

    cuisine_df = cuisine_df[
        cuisine_df['Price range'].between(
            max(1, selected_price - 1),
            min(4, selected_price + 1)
        )
    ]

    # Sort Restaurants

    cuisine_df = cuisine_df.sort_values(
        by=[
            'Aggregate rating',
            'Votes'
        ],
        ascending=False
    )

    st.subheader("Recommended Restaurants")

    # Empty Handling

    if cuisine_df.empty:

        st.warning(
            "No restaurants found for selected filters."
        )

    else:

        st.dataframe(
            cuisine_df[[
                'Restaurant Name',
                'City',
                'Cuisines',
                'Aggregate rating',
                'Average Cost for two',
                'Votes'
            ]].head(20),
            width='stretch'
        )

        # Restaurant Map

        st.subheader("Restaurant Locations")

        map_df = cuisine_df[[
            'Latitude',
            'Longitude'
        ]].dropna()

        map_df = map_df.rename(columns={
            'Latitude': 'lat',
            'Longitude': 'lon'
        })

        st.map(map_df)

# Location Analysis

elif page == "Location Analysis":

    st.title("Location-Based Analysis")

    # City Selection

    city_analysis = st.selectbox(
        "Select City",
        sorted(df['City'].dropna().unique())
    )

    # Filter City Data

    city_df = df[
        df['City'] == city_analysis
    ]

    # Average Coordinates

    avg_lat = city_df['Latitude'].mean()

    avg_lon = city_df['Longitude'].mean()

    # Create Folium Map

    restaurant_map = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=12
    )

    # Add Restaurant Markers

    for _, row in city_df.iterrows():

        if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']):

            popup_text = f"""
            <b>{row['Restaurant Name']}</b><br>
            Cuisine: {row['Cuisines']}<br>
            Rating: {row['Aggregate rating']}<br>
            Cost for Two: {row['Average Cost for two']}<br>
            Votes: {row['Votes']}
            """

            folium.Marker(
                location=[
                    row['Latitude'],
                    row['Longitude']
                ],
                popup=popup_text
            ).add_to(restaurant_map)

    # Display Interactive Map

    st.subheader("Interactive Restaurant Map")

    st_folium(
        restaurant_map,
        width=1200,
        height=600
    )

    # Average Rating

    st.subheader("Average Rating")

    avg_city_rating = round(
        city_df['Aggregate rating'].mean(),
        2
    )

    st.metric(
        "Average Rating",
        avg_city_rating
    )

    # Price Range Distribution

    st.subheader("Price Range Distribution")

    fig4 = px.histogram(
        city_df,
        x='Price range'
    )

    st.plotly_chart(
        fig4,
        width='stretch'
    )

    # Top Cuisines

    st.subheader("Top Cuisines")

    city_cuisines = (
        city_df['Cuisines']
        .str.split(',')
        .explode()
        .str.strip()
    )

    top_city_cuisines = (
        city_cuisines
        .value_counts()
        .head(10)
    )

    fig5 = px.bar(
        x=top_city_cuisines.index,
        y=top_city_cuisines.values
    )

    st.plotly_chart(
        fig5,
        width='stretch'
    )
    # Cuisine Distribution

    st.subheader("Top Cuisines")

    city_cuisines = (
        city_df['Cuisines']
        .str.split(',')
        .explode()
        .str.strip()
    )

    top_city_cuisines = (
        city_cuisines
        .value_counts()
        .head(10)
    )

    fig5 = px.bar(
        x=top_city_cuisines.index,
        y=top_city_cuisines.values
    )

    st.plotly_chart(
        fig5,
        width='stretch'
    )