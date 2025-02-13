import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
import pydeck as pdk
import httpx
import asyncio
import numpy as np
from typing import List, Dict, Any

# Function to convert the CSV data to JSON


def csv_to_json(csv_data: pd.DataFrame) -> List[Dict[str, Any]]:
    markers_data = []
    for index, row in csv_data.iterrows():
        # Set default values to ensure compliance with API requirements
        type_value = row['TYPE'] if pd.notna(
            row['TYPE']) else "INCIDENT_LOCATION_LOW_PRIORITY"
        description_value = row['DESCRIPTION'] if pd.notna(
            row['DESCRIPTION']) else f"Incident {index}"
        event_time_value = (datetime.now(
        ) + timedelta(seconds=row['DELAY'] if pd.notna(row['DELAY']) else 0)).isoformat()
        latitude_value = row['LATITUDE']
        longitude_value = row['LONGITUDE']

        # Create the marker dict
        marker = {
            "type": type_value,
            "description": description_value,
            "event_time": event_time_value,
            "latitude": latitude_value,
            "longitude": longitude_value,
        }

        # Optional fields: only add if not null or empty
        if pd.notna(row['EXTERNALID']) and row['EXTERNALID']:
            marker["external_id"] = row['EXTERNALID']
        if pd.notna(row['UUID']) and row['UUID']:
            marker["uuid"] = row['UUID']

        markers_data.append(marker)
    return markers_data


def main():
    # Streamlit app interface
    st.title("Marker Generator")

    with st.expander("Configuration", expanded=False):
        api_url = st.text_input("API URL", type="default",
                                value=st.query_params.get("api_url", "https://api.skydio.com"))
        api_token = st.text_input("API Token", type="password",
                                  value=st.query_params.get("api_token"))

    st.markdown("""
    #### Usage:

    1. First make sure you have a corresponding API Key to use with [Skydio Markers API](https://apidocs.skydio.com/reference/87u7ko6t0dqmfgngmarkers_post_v0_marker)
    2. Utilize [this Google Sheets](https://docs.google.com/spreadsheets/d/1Iz7aVcoIcEGnVnqyHDeo-MC9nO6ORBlwpj7QSgHHfVs/edit?gid=0#gid=0) template to create your markers
    3. "File" > "Download" > "Comma-separated values (.csv, current sheet)"
    4. [Optional] Verify data within this app through tabular viewer or map
    5. Provide API Key + upload!

    Then upload file below:""")

    # Upload CSV File
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        # Read CSV into a pandas DataFrame
        csv_data = pd.read_csv(uploaded_file)
        csv_data.sort_values(by='DELAY', ascending=True)

        # Collapsible section for the uploaded CSV (open by default)
        with st.expander("Uploaded CSV Data", expanded=True):
            st.subheader("Uploaded CSV File:")
            st.dataframe(csv_data)

        # Convert CSV to JSON
        markers_json = csv_to_json(csv_data)

        # Collapsible section for the generated JSON (collapsed by default)
        with st.expander("Generated Markers JSON", expanded=False):
            st.subheader("Generated JSON:")
            st.json(markers_json)  # Display JSON in the Streamlit app

            # Log JSON to the Streamlit app console
            st.write("Log JSON:")
            st.code(json.dumps(markers_json, indent=2),
                    language='json')  # Log JSON to the page

            # Option to download the JSON as a file
            json_string = json.dumps(markers_json, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_string,
                file_name="markers.json",
                mime="application/json"
            )

        with st.expander("Markers Map", expanded=False):
            # Ensure the latitude and longitude columns are numeric
            csv_data['LATITUDE'] = pd.to_numeric(
                csv_data['LATITUDE'], errors='coerce')
            csv_data['LONGITUDE'] = pd.to_numeric(
                csv_data['LONGITUDE'], errors='coerce')

            # Drop rows with invalid coordinates
            csv_data = csv_data.dropna(subset=['LATITUDE', 'LONGITUDE'])

            # Assign color based on type
            def get_color(marker_type):
                if "CRITICAL" in marker_type:
                    return [255, 0, 0, 160]  # Red for critical
                elif "HIGH" in marker_type or "SERIOUS" in marker_type:
                    return [255, 165, 0, 160]  # Orange for serious
                else:
                    return [255, 255, 0, 160]  # Yellow for non-serious

            csv_data['COLOR'] = csv_data['TYPE'].apply(get_color)

            # Create a PyDeck map layer with circular markers
            layer = pdk.Layer(
                'ScatterplotLayer',
                data=csv_data,
                get_position='[LONGITUDE, LATITUDE]',
                get_color='COLOR',
                get_radius=1,
                radius_min_pixels=10,
                radius_max_pixels=10,
                pickable=True
            )

            # Example csv_data with latitude and longitude columns
            min_latitude = csv_data['LATITUDE'].min()
            max_latitude = csv_data['LATITUDE'].max()
            min_longitude = csv_data['LONGITUDE'].min()
            max_longitude = csv_data['LONGITUDE'].max()

            # Compute the center point (mean latitude and longitude)
            center_latitude = csv_data['LATITUDE'].mean()
            center_longitude = csv_data['LONGITUDE'].mean()

            # Compute the max distance in latitude and longitude (rough approximation)
            lat_diff = max_latitude - min_latitude
            lon_diff = max_longitude - min_longitude

            # Approximate zoom level based on distance (this is a rough calculation)
            # Higher zoom values give closer views; smaller values zoom out more
            zoom_level = 8 - np.log(max(lat_diff, lon_diff))

            # Set up the view state in pydeck
            view_state = pdk.ViewState(
                latitude=center_latitude,
                longitude=center_longitude,
                zoom=zoom_level,
                pitch=0  # Top-down view
            )

            # Create the PyDeck map with Mapbox base style
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "{DESCRIPTION}"},
                map_style="mapbox://styles/mapbox/streets-v11"  # Mapbox base style
            )

            # Display the map in Streamlit
            st.pydeck_chart(deck)

        async def send_marker(session, request_url, headers, marker, i):
            delay = (datetime.fromisoformat(
                marker["event_time"]) - datetime.now()).total_seconds()
            if delay > 0:
                await asyncio.sleep(delay)
            response = await session.post(request_url, headers=headers, json=marker)

            # Uncomment below to debug: log request and response details
            # st.write(f"Marker {i} payload: {json.dumps(marker, indent=2)}")
            # st.write(f"Marker {i} response: {response.status_code} - {response.text}")

            if response.status_code != 200:
                return f"Marker {i}: {response.text}"
            return None

        async def send_markers():
            request_url = f"{api_url}/api/v0/marker"

            headers = {
                "Authorization": f"{api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            error_messages = []

            async with httpx.AsyncClient() as session:
                tasks = []
                for i, marker in enumerate(markers_json):
                    tasks.append(send_marker(session, request_url,
                                             headers, marker, i))
                results = await asyncio.gather(*tasks)

            for result in results:
                if result:
                    error_messages.append(result)

            if error_messages:
                st.error("Errors occurred during the upload:\n" +
                         "\n".join(error_messages))
            else:
                st.success("All markers uploaded successfully!")

        if st.button("Send Markers"):
            asyncio.run(send_markers())


main()
