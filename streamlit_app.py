import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
import pydeck as pdk
import httpx
import asyncio
import io
from typing import List, Dict, Any

# Function to convert the CSV data to JSON


def csv_to_json(csv_data: pd.DataFrame) -> List[Dict[Any, Any]]:
    markers_data = []
    for index, row in csv_data.iterrows():
        marker = {
            "type": row['TYPE'] or "INCIDENT_LOCATION_LOW_PRIORITY",
            "description": row['DESCRIPTION'] or f"Incident {index}",
            "event_time": (datetime.now() + timedelta(seconds=row['DELAY'] or 0)).isoformat(),
            "external_id": None if pd.isna(row['EXTERNALID']) or not row['EXTERNALID'] else row['EXTERNALID'],
            "latitude": row['LATITUDE'],
            "longitude": row['LONGITUDE'],
            "uuid": None if pd.isna(row['UUID']) or not row['UUID'] else row['UUID']
        }
        markers_data.append(marker)
    return markers_data


# Streamlit app interface
st.title("CSV to JSON Marker Conversion, Mapping, and Upload")

st.markdown("""
Use this as a template: 

https://docs.google.com/spreadsheets/d/1Iz7aVcoIcEGnVnqyHDeo-MC9nO6ORBlwpj7QSgHHfVs/edit?gid=0#gid=0

Export as `.csv`

Then upload file here:""")

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

        # Create a PyDeck map layer
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=csv_data,
            get_position='[LONGITUDE, LATITUDE]',
            get_color='[200, 30, 0, 160]',
            get_radius=100,
            pickable=True
        )

        # Set the view for the map (centered around the first marker)
        view_state = pdk.ViewState(
            latitude=csv_data['LATITUDE'].mean(),
            longitude=csv_data['LONGITUDE'].mean(),
            zoom=12,
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

    # url = "https://api.skydio.com/api/v0/marker"

    # payload = {
    #     "type": "INCIDENT_LOCATION_LOW_PRIORITY",
    #     "description": "Big problem",
    #     "event_time": "2022-05-03T03:10:52.503+00:00",
    #     "external_id": "XYZ-32939",
    #     "latitude": 45,
    #     "longitude": 10,
    #     "uuid": "9565d76b-ca6a-40db-8486-7aa1bbc59cc0"
    # }
    # headers = {
    #     "accept": "application/json",
    #     "content-type": "application/json"
    # }

    api_url = st.text_input("API url", type="default",
                            value="https://api.skydio.com")
    api_key = st.text_input("API key", type="password")

    async def upload_marker(session, request_url, headers, marker, i):
        delay = (datetime.fromisoformat(
            marker["event_time"]) - datetime.now()).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)
        response = await session.post(request_url, headers=headers, data=marker)
        if response.status_code != 200:
            return f"Marker {i}: {response.text}"
        return None

    async def send_markers():
        request_url = f"{api_url}/api/v0/marker"

        headers = {
            "Authorization": f"{api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        error_messages = []

        async with httpx.AsyncClient() as session:
            tasks = []
            for i, marker in enumerate(markers_json):
                tasks.append(upload_marker(session, request_url,
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
