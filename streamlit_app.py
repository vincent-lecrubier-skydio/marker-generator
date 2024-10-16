import streamlit as st
import pandas as pd
import json
from datetime import datetime
import pydeck as pdk
# Assuming you have this for the upload functionality
import marker_upsert_utils as marker_utils

# Function to convert the CSV data to JSON


def csv_to_json(csv_data):
    markers_data = {
        "data": {
            "organization": {
                "markers": {
                    "edges": []
                }
            }
        }
    }

    for index, row in csv_data.iterrows():
        # Build the marker JSON structure based on the CSV structure
        # Format time in the required format
        event_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        edge = {
            "node": {
                "jsonMetadata": {
                    # Incident description
                    "description": row['INCIDENT_DESCRIPTION'],
                    "eventTime": event_time,  # Current timestamp in ISO format with 'Z' for UTC
                    "latitude": row['LATITUDE'],  # Latitude
                    "longitude": row['LONGITUDE']  # Longitude
                },
                "type": row['INCIDENT_TYPE']  # Marker type
            }
        }
        markers_data["data"]["organization"]["markers"]["edges"].append(edge)

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

    # Pydeck map to visualize the lat/long from the CSV data
    st.subheader("Map of Incident Locations")

    # Ensure the latitude and longitude columns are numeric
    csv_data['LATITUDE'] = pd.to_numeric(csv_data['LATITUDE'], errors='coerce')
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
        tooltip={"text": "{INCIDENT_DESCRIPTION}"},
        map_style="mapbox://styles/mapbox/streets-v11"  # Mapbox base style
    )

    # Display the map in Streamlit
    st.pydeck_chart(deck)

# API credentials and details
st.subheader("API Interaction")
org_uuid = st.text_input("Organization UUID:", value="ORG_UUID")
api_key = st.text_input("API Key:", type="password")
api_url = st.text_input(
    "API URL:", value="https://your-api-endpoint.com/graphql")

# Upload Markers Button
if st.button("Upload Markers"):
    if org_uuid and api_key and api_url:
        try:
            # Extract the list of markers from the nested structure
            edges = markers_json["data"]["organization"]["markers"]["edges"]

            # Flatten the structure to get a list of markers (each "node" from the edges)
            markers_list = [edge["node"] for edge in edges]

            # Rename 'jsonMetadata' to 'json_metadata' for each marker
            for marker in markers_list:
                marker["json_metadata"] = marker.pop("jsonMetadata")

            # Debugging: Check the type and structure of markers_list before uploading
            st.write(f"Markers List Type: {type(markers_list)}")
            st.write(f"Markers List Content: {markers_list}")

            # Pass the list of markers to upsert_markers function
            created_count = marker_utils.upsert_markers(
                api_url, org_uuid, api_key, markers_list)
            st.success(f"Successfully uploaded {created_count} markers.")
        except Exception as e:
            st.error(f"Error uploading markers: {str(e)}")
    else:
        st.error("Please provide the Organization UUID, API Key, and API URL.")

# Delete Markers Button
if st.button("Delete All Markers"):
    if org_uuid and api_key and api_url:
        try:
            # Call the function to clear all markers and operations
            response = marker_utils.clear_markers_and_operations(
                api_url, org_uuid, api_key)
            st.success(
                f"Successfully deleted {response['data']['clearOrgOperationsAndMarkers']['markersDeleted']} markers.")
        except Exception as e:
            st.error(f"Error deleting markers: {str(e)}")
    else:
        st.error("Please provide the Organization UUID, API Key, and API URL.")
