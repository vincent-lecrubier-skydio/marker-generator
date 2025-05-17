import streamlit as st
import pandas as pd
import json
import os
import openai 
from openai import OpenAI 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
from io import StringIO
from datetime import datetime, timedelta
import pydeck as pdk
import httpx
import openai
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple
from mapbox_util import forward_geocode
from real_agencies import get_random_agency
import re

def render_gpt_output(result: str):
    import re

    st.markdown("### 🤖 AI Overview")

    # Extract main content blocks
    anchor = re.search(r"\*\*RESPONSE ANCHOR\*\*\s*(.*?)\n\n", result, re.DOTALL)
    overview = re.search(r"\*\*AGENCY OVERVIEW\*\*\s*(.*?)\n\n", result, re.DOTALL)
    csv_block = re.search(r"```csv\s*\n(.*?)```", result, re.DOTALL)

    # Location section
    if anchor:
        st.markdown("#### 📍 Location")
        st.markdown(anchor.group(1).strip())

    # Agency overview with inline stat highlighting
    if overview:
        st.markdown("#### 🏢 Agency")

        full_text = overview.group(1).strip()

        # Inline highlighting of structured data
        full_text = re.sub(r"(\d{2,5})\s+sworn\s+(?:officers|personnel)", r"**\1 sworn officers**", full_text, flags=re.IGNORECASE)
        full_text = re.sub(r"(?:population|pop\.?)\s+(?:served|covered).*?(\d[\d,]*)", r"**Population: \1**", full_text, flags=re.IGNORECASE)
        full_text = re.sub(r"(?:land area|covers|serves).*?(\d{1,5}) ?(?:sq\.?|square)? ?mi", r"**Land Area: \1 sq mi**", full_text, flags=re.IGNORECASE)
        full_text = re.sub(r"(drone program|uas).*?(yes|no|exploring|unknown|not reported)", r"**Drone Program: \2**", full_text, flags=re.IGNORECASE)

        st.markdown(full_text)

    # CSV incident output
    if csv_block:
        st.markdown("#### 📄 Incidents")
        st.code(csv_block.group(1).strip(), language="csv")



def csv_to_json(
    csv_data: pd.DataFrame, force_new_markers_ui: bool = False
) -> List[Dict[str, Any]]:
    markers_data = []
    for index, row in csv_data.iterrows():
        # Set default values to ensure compliance with API requirements
        type_value = (
            row["TYPE"] if pd.notna(
                row["TYPE"]) else "INCIDENT_LOCATION_LOW_PRIORITY"
        )
        description_value = (
            row["DESCRIPTION"] if pd.notna(
                row["DESCRIPTION"]) else f"Incident {index}"
        )
        event_time_value = (
            datetime.now()
            + timedelta(seconds=row["DELAY"] if pd.notna(row["DELAY"]) else 0)
        ).isoformat()
        details_value = (
            row["DETAILS"]
            if "DETAILS" in row.keys() and pd.notna(row["DETAILS"])
            else f"Example details {index}"
        )
        latitude_value = row["LATITUDE"]
        longitude_value = row["LONGITUDE"]

        # Create the marker dict
        if not force_new_markers_ui:
            marker = {
                "type": type_value,
                "description": description_value,
                "event_time": event_time_value,
                "latitude": latitude_value,
                "longitude": longitude_value,
            }
        else:
            marker = {
                "type": "INCIDENT",
                "title": description_value,
                "description": details_value,
                "event_time": event_time_value,
                "latitude": latitude_value,
                "longitude": longitude_value,
            }

        # Optional fields: only add if not null or empty
        # NOTE: Checking some fields with "in row.keys()" to avoid KeyError and make the code backward compatible
        if pd.notna(row["EXTERNALID"]) and row["EXTERNALID"]:
            marker["external_id"] = row["EXTERNALID"]
        if "UUID" in row.keys() and pd.notna(row["UUID"]) and row["UUID"]:
            marker["uuid"] = row["UUID"]
        if "AREA" in row.keys() and pd.notna(row["AREA"]) and row["AREA"]:
            marker["area"] = row["AREA"]
        if "TITLE" in row.keys() and pd.notna(row["TITLE"]) and row["TITLE"]:
            marker["title"] = row["TITLE"]

        marker_details = {}
        if "CODE" in row.keys() and pd.notna(row["CODE"]) and row["CODE"]:
            marker_details["code"] = row["CODE"]
        if (
            "INCIDENT_ID" in row.keys()
            and pd.notna(row["INCIDENT_ID"])
            and row["INCIDENT_ID"]
        ):
            marker_details["incident_id"] = row["INCIDENT_ID"]
        if "PRIORITY" in row.keys() and pd.notna(row["PRIORITY"]) and row["PRIORITY"]:
            marker_details["priority"] = row["PRIORITY"]

        if marker_details:
            marker["marker_details"] = marker_details

        markers_data.append(marker)
    return markers_data


def format_scenario_mode(mode):
    if mode == "preset":
        return "⚡️ Preset"
    if mode == "random":
        return "🎲 Random"
    if mode == "custom":
        return "📄 Custom"
    if mode == "precision_drop":
        return "📍 Precision Drop"
    return "preset"


def parse_lat_lon(s):
    pattern = r'[-+]?\d*\.\d+|\d+'
    matches = re.findall(pattern, s)
    if len(matches) == 2:
        return float(matches[0]), float(matches[1])
    return None


def main():
    st.set_page_config(page_title="Marker Generator",
                       page_icon="🚨", layout="wide")

    st.title("🚨 Marker Generator")

    csv_data = None

    force_new_markers_ui = False

    if "mode" not in st.session_state:
        if "mode" in st.query_params:
            mode_param = st.query_params.get("mode")
            if mode_param in ["preset", "random", "custom", "precision_drop"]:
                st.session_state["mode"] = mode_param
            else:
                st.session_state["mode"] = "preset"
        else:
            st.session_state["mode"] = "preset"

    mode = st.segmented_control(
        "Mode:",
        ["preset", "random", "custom", "precision_drop"],
        format_func=format_scenario_mode,
        key="mode"
    )
    if mode is not None:
        st.query_params["mode"] = mode
    elif "mode" in st.query_params:
        del st.query_params["mode"]

    if mode == "preset":
        directory = "./scenarios"  # Change this to the desired rdirectory path
        scenario_files = [
            os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith(".csv")
        ]
        scenario_files.sort()

        if "scenario" not in st.session_state:
            if "scenario" in st.query_params:
                scenario_param = st.query_params.get("scenario")
                if scenario_param in scenario_files:
                    st.session_state["scenario"] = scenario_param
                elif scenario_param.strip() in scenario_files:
                    st.session_state["scenario"] = scenario_param.strip()
                elif scenario_param == "clearview-demo-1":
                    st.session_state["scenario"] = "1 - Public Safety DFR - HQ Demo"
                elif scenario_param == "clearview-demo-2":
                    st.session_state["scenario"] = "1 - Public Safety DFR - HQ Demo"
                else:
                    st.session_state["scenario"] = None
            else:
                st.session_state["scenario"] = None
        scenario_preset_label = (
            "🔴 Select Preset Scenario"
            if st.session_state.get("scenario") is None
            else f"🟢 Select Preset Scenario: {st.session_state.get('scenario')}"
        )
        with st.expander(
            scenario_preset_label, expanded=st.session_state.get(
                "scenario") is None
        ):
            st.markdown("""
            - Select one of the preset scenarios below
            - You can request additional presets to be added by asking Vincent Lecrubier at Skydio""")
            scenario = st.pills("Presets: ", scenario_files, key="scenario")
            if scenario is not None:
                st.query_params["scenario"] = scenario
            elif "scenario" in st.query_params:
                del st.query_params["scenario"]
            scenario_file = (
                f"{directory}/{scenario}.csv" if scenario is not None else None
            )
            if scenario_file:
                csv_data = pd.read_csv(scenario_file)

    if mode == "random":
        sample_csv_data = None
        directory = "./samples"  # Change this to the desired directory path
        sample_files = [os.path.splitext(f)[0] for f in os.listdir(
            directory) if f.endswith(".csv")]
        sample_files.sort()

        if "sample" not in st.session_state:
            if "sample" in st.query_params:
                sample_param = st.query_params.get("sample")
                if sample_param in sample_files:
                    st.session_state["sample"] = sample_param
                else:
                    st.session_state["sample"] = None

        if "location" not in st.session_state:
            if "location" in st.query_params:
                location_param = st.query_params.get("location")
                st.session_state["location"] = location_param
            else:
                st.session_state["location"] = "3000 Clearview way, San Mateo, CA"

        scenario_random_label = (
            "🔴 Generate Random Scenario"
            if st.session_state.get("sample") is None
            else f"🟢 Generate Random Scenario: {st.session_state.get('sample')}"
        )
        with st.expander(scenario_random_label, expanded=st.session_state.get(
                "location") is None or st.session_state.get(
                "sample") is None):
            st.markdown("""
              - Select one of the categories below to generate random markers
              - Enter a location around which markers will appear
              - Customize number of markers, radius and delay parameters if needed
            """)
            sample = st.pills(
                "Category: ", sample_files, key="sample")
            if sample is not None:
                st.query_params["sample"] = sample
            elif "sample" in st.query_params:
                del st.query_params["sample"]
            scenario_file = f"{directory}/{sample}.csv" if sample is not None else None
            if scenario_file:
                sample_csv_data = pd.read_csv(scenario_file)

            location = st.text_input(
                "Location Address or Coordinates (lat,lon)", key="location")

            coords = forward_geocode(location)

            center_lat = None
            center_lon = None
            if coords:
                center_lat = coords[1]
                center_lon = coords[0]
            else:
                lat_lon = parse_lat_lon(location)
                if lat_lon:
                    center_lat, center_lon = lat_lon
            if center_lat is None or center_lon is None:
                st.error(
                    "Invalid location. Please enter a valid address or coordinates (lat,lon)")
                del st.query_params["location"]
                return
            else:
                st.query_params["location"] = location

            st.markdown(f"""
                Location Coordinates:

                ```python
                {center_lat}, {center_lon}
                ```
                """)

            radius_mi = st.number_input(
                "Radius around location (mi)", value=1.0, step=0.1)
            sample_size = st.number_input(
                "Number of markers to generate", min_value=1, max_value=1000, value=10, step=1)
            min_delay = st.number_input(
                "Minimum Delay (Seconds between click 'Send markers' and Start of scenario)", value=0, step=1)
            max_delay = st.number_input(
                "Maximum Delay (Seconds between click 'Send markers' and End of scenario)", value=10, step=1)

        if sample_csv_data is not None:
            # Randomize selected markers
            csv_data = sample_csv_data.sample(n=sample_size)

            random_indices = csv_data.index
            for idx in random_indices:
                r = radius_mi * 1.60934 * np.sqrt(np.random.uniform(0, 1))
                theta = np.random.uniform(0, 2 * np.pi)
                delta_lat = (r * np.cos(theta)) / 111.32
                delta_lon = (r * np.sin(theta)) / \
                    (111.32 * np.cos(np.deg2rad(center_lat)))
                csv_data.loc[idx, "LATITUDE"] = center_lat + delta_lat
                csv_data.loc[idx, "LONGITUDE"] = center_lon + delta_lon
                random_delay = np.random.uniform(min_delay, max_delay)
                csv_data.loc[idx, "DELAY"] = int(random_delay)
            csv_data.sort_values(
                by="DELAY", ascending=True, inplace=True)

    if mode == "custom":
        scenario_upload_label = (
            "🔴 Upload Custom Scenario"
            if st.session_state.get("scenario_uploaded_file") is None
            else "🟢 Upload Custom Scenario: Ok"
        )
        with st.expander(
            scenario_upload_label,
            expanded=True,
        ):
            st.markdown("""
            - Utilize [this Google Sheets](https://docs.google.com/spreadsheets/d/1Iz7aVcoIcEGnVnqyHDeo-MC9nO6ORBlwpj7QSgHHfVs/edit?gid=0#gid=0) template to create your markers
            - "File" > "Download" > "Comma-separated values (.csv, current sheet)"
            - Upload CSV file below""")
            uploaded_file = st.file_uploader(
                "", type=["csv"], key="scenario_uploaded_file"
            )
            force_new_markers_ui = st.toggle(
                "Convert old .csv files automatically to the new markers UI. **Note: In the new UI all incident markers are critical red color.**",
                key="force_new_markers",
            )
            scenario_file = uploaded_file
            if scenario_file:
                csv_data = pd.read_csv(scenario_file)

    if mode == "precision_drop":
        st.markdown("<p style='font-size:16px; color:gray;'>Generate ultra-realistic incident markers for any public safety agency — with real addresses, local relevance, and drone-suited scenarios. </p>", unsafe_allow_html=True)

        st.markdown("### Select an agency to generate markers for:")

        # Randomize first
        if st.button("🎲 Pick Random Agency"):
            st.session_state["agency_name"] = get_random_agency()
            st.rerun()


        # Then display the input
        agency_name = st.text_input(
            "### Enter an Agency:",
            key="agency_name"
        )

        anchor_location = st.text_input("OPTIONAL: Pre-select your dock Location (address or lat/long)", "")

        scenario_upload_label = (
            f"🟢" if agency_name.strip()
            else "🔴 Enter a Public Safety Agency to generate markers"
        )

        with st.expander(scenario_upload_label, expanded=True):
            if st.button("## 🧠 Generate Markers via OpenAI"):
                with st.spinner("Generating 10 markers in a ~2mi radius to dock"):
                    try:
                        # Load system prompt
                        with open("source_text.txt", "r", encoding="utf-8") as f:
                            system_prompt = f.read()

                        # Construct user prompt
                        user_prompt = f"Agency: {agency_name}"
                        if anchor_location.strip():
                            user_prompt += f"\nAnchor Location: {anchor_location.strip()}"

                        # Call GPT
                        response = client.chat.completions.create(
                            model="gpt-4o",  
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.3
                        )

                        result = response.choices[0].message.content
                        render_gpt_output(result)

                        # Extract CSV from code block
                        import re
                        match = re.search(r"```csv\s*\n(.*?)```", result, re.DOTALL | re.IGNORECASE)
                        if not match:
                            match = re.search(r"```\s*\n(.*?)```", result, re.DOTALL)

                        if not match:
                            raise ValueError("Could not find a valid CSV code block in GPT output.")

                        csv_text = match.group(1).strip()
                        st.session_state["tailored_pasted_csv"] = csv_text  # optional: keep for manual edits

                        # ✅ Parse CSV directly into DataFrame
                        from io import StringIO
                        csv_data = pd.read_csv(StringIO(csv_text))
                        st.success("CSV loaded successfully!")

                        # ✅ Geocode if needed
                        if "LATITUDE" not in csv_data.columns and "ADDRESS" in csv_data.columns:
                            with st.spinner("Converting addresses to coordinates..."):
                                lat_list = []
                                lon_list = []
                                for address in csv_data["ADDRESS"]:
                                    coords = forward_geocode(address)
                                    if coords:
                                        lon, lat = coords
                                    else:
                                        lon, lat = None, None
                                    lat_list.append(lat)
                                    lon_list.append(lon)
                                csv_data["LATITUDE"] = lat_list
                                csv_data["LONGITUDE"] = lon_list
                                csv_data.dropna(subset=["LATITUDE", "LONGITUDE"], inplace=True)


                    except Exception as e:
                        st.error(f"Failed to generate scenario: {e}")

    if "api_url" not in st.session_state:
        if "api_url" in st.query_params:
            st.session_state["api_url"] = st.query_params.get("api_url")
        else:
            st.session_state["api_url"] = "https://api.skydio.com"
    if "api_token" not in st.session_state:
        if "api_token" in st.query_params:
            st.session_state["api_token"] = st.query_params.get("api_token")
        else:
            st.session_state["api_token"] = None

    configuration_label = (
        "🔴 API Configuration"
        if st.session_state.get("api_url") is None
        or st.session_state.get("api_token") is None
        else "🟢 API Configuration"
    )
    with st.expander(configuration_label):
        api_url = st.text_input("API URL", type="default", key="api_url")
        api_token = st.text_input(
            "API Token", type="password", key="api_token")
        if api_url is not None and api_url != "https://api.skydio.com":
            st.query_params["api_url"] = api_url
        elif "api_url" in st.query_params:
            del st.query_params["api_url"]
        if api_token is not None:
            st.query_params["api_token"] = api_token
        elif "api_token" in st.query_params:
            del st.query_params["api_token"]
            
    # Add marker management tools in a dedicated section
    with st.expander("🗑️ Marker Management", expanded=False):
        st.markdown("### Delete Latest Markers - AXON CEO SUMMIT ONLY CAUTION DO NOT TOUCH")
        st.markdown("Use this tool to delete the most recent markers from the API. This action cannot be undone.")
        
        # Hard-code the number of markers to delete to 25
        st.write("This button will delete the 25 most recent markers from the API.")
        st.markdown("**⚠️ Warning: This action cannot be undone.**")
        
        async def fetch_latest_markers(limit: int) -> Tuple[List[Dict[str, Any]], List[str]]:
            """Fetch the latest markers from the API"""
            request_url = f"{st.session_state.api_url}/api/v0/markers?limit={limit}"
            
            headers = {
                "Authorization": f"{st.session_state.api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            
            error_messages = []
            markers = []
            
            try:
                async with httpx.AsyncClient() as session:
                    response = await session.get(request_url, headers=headers)
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("data") and "markers" in result["data"]:
                            markers = result["data"]["markers"]
                        else:
                            error_messages.append("No markers found in the response")
                    else:
                        error_messages.append(f"Error fetching markers: {response.text}")
            except Exception as e:
                error_messages.append(f"Exception while fetching markers: {str(e)}")
                
            return markers, error_messages
        
        async def delete_marker(session, marker_id: str, headers: Dict[str, str]) -> str:
            """Delete a single marker and return any error message"""
            request_url = f"{st.session_state.api_url}/api/v0/marker/{marker_id}/delete"
            try:
                response = await session.delete(request_url, headers=headers)
                if response.status_code != 200:
                    return f"Error deleting marker {marker_id}: {response.text}"
                return ""
            except Exception as e:
                return f"Exception while deleting marker {marker_id}: {str(e)}"
        
        async def delete_markers_in_batches(markers: List[Dict[str, Any]], batch_size: int = 5, delay_ms: int = 50) -> Tuple[List[str], int]:
            """Delete markers in batches to avoid overwhelming the API"""
            headers = {
                "Authorization": f"{st.session_state.api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            
            error_messages = []
            success_count = 0
            
            async with httpx.AsyncClient() as session:
                for i in range(0, len(markers), batch_size):
                    # Process markers in batches
                    batch = markers[i:i+batch_size]
                    batch_tasks = []
                    
                    for marker in batch:
                        if "uuid" in marker and marker["uuid"]:
                            batch_tasks.append(delete_marker(session, marker["uuid"], headers))
                    
                    # Process each batch concurrently
                    batch_results = await asyncio.gather(*batch_tasks)
                    
                    # Count successes and collect errors
                    for result in batch_results:
                        if result:
                            error_messages.append(result)
                        else:
                            success_count += 1
                    
                    # Small delay between batches to avoid overwhelming the API
                    if i + batch_size < len(markers):
                        await asyncio.sleep(delay_ms / 1000)
                    
                    # Update progress
                    progress_text = f"Deleted {success_count} of {len(markers)} markers..."
                    st.session_state.delete_progress.progress((i + len(batch)) / len(markers), text=progress_text)
            
            return error_messages, success_count
            
        async def delete_latest_markers(limit_count: int):
            """Fetch and delete the latest markers"""
            # Initialize progress bar
            if 'delete_progress' not in st.session_state:
                st.session_state.delete_progress = st.progress(0, text="Fetching latest markers...")
            else:
                st.session_state.delete_progress.progress(0, text="Fetching latest markers...")
                
            # Use the directly passed limit value instead of session state
            limit = limit_count
            batch_size = min(5, limit)  # Use smaller batches for larger limits
            
            # Step 1: Fetch the latest markers with explicit limit
            st.write(f"Fetching {limit} latest markers...")
            markers, fetch_errors = await fetch_latest_markers(limit=limit)
            
            if fetch_errors:
                for error in fetch_errors:
                    st.error(error)
                st.session_state.delete_progress.progress(1.0, text="Failed to fetch markers")
                return
            
            if not markers:
                st.warning("No markers found to delete")
                st.session_state.delete_progress.progress(1.0, text="No markers found")
                return
                
            st.session_state.delete_progress.progress(0.2, text=f"Found {len(markers)} markers to delete")
            
            # Step 2: Delete the markers in batches
            delete_errors, success_count = await delete_markers_in_batches(markers, batch_size)
            
            # Step 3: Display results
            if delete_errors:
                st.error(f"Deleted {success_count} markers, but encountered {len(delete_errors)} errors:")
                for error in delete_errors[:5]:  # Show first 5 errors
                    st.error(error)
                if len(delete_errors) > 5:
                    st.error(f"... and {len(delete_errors) - 5} more errors (not shown)")
                    
                if any("Invalid Authorization header" in s for s in delete_errors):
                    st.info("Advice: Please check your API Token and make sure it is correct.")
            else:
                st.success(f"Successfully deleted {success_count} markers!")
                
            st.session_state.delete_progress.progress(1.0, text=f"Completed: Deleted {success_count} markers")

        # Simple button to delete exactly 25 markers
        if st.button("🗑️ Delete Latest 25 Markers"):
            # Always use 25 as the limit
            asyncio.run(delete_latest_markers(25))

    if csv_data is not None:
        with st.expander("📄 Toolbox", expanded=False):
            # Read CSV into a pandas DataFrame
            csv_data.sort_values(by="DELAY", ascending=True)

            markers_map_viewer, scenario_editor, markers_json_viewer = st.tabs(
                [
                    "🗺️ Markers Map",
                    "✏️ Scenario Editor",
                    "👾 Markers JSON",
                ]
            )

            # Collapsible section for the uploaded CSV (open by default)
            with scenario_editor:
                st.data_editor(csv_data)

            # Convert CSV to JSON
            markers_json = csv_to_json(csv_data, force_new_markers_ui)

            # Collapsible section for the generated JSON (collapsed by default)
            with markers_json_viewer:
                st.json(markers_json)  # Display JSON in the Streamlit app

                # Log JSON to the Streamlit app console
                st.write("Log JSON:")
                st.code(
                    json.dumps(markers_json, indent=2), language="json"
                )  # Log JSON to the page

                # Option to download the JSON as a file
                json_string = json.dumps(markers_json, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_string,
                    file_name="markers.json",
                    mime="application/json",
                )

            with markers_map_viewer:
                # Ensure the latitude and longitude columns are numeric
                csv_data["LATITUDE"] = pd.to_numeric(
                    csv_data["LATITUDE"], errors="coerce"
                )
                csv_data["LONGITUDE"] = pd.to_numeric(
                    csv_data["LONGITUDE"], errors="coerce"
                )

                # Drop rows with invalid coordinates
                csv_data = csv_data.dropna(subset=["LATITUDE", "LONGITUDE"])

                # Assign color based on type
                def get_color(marker_type):
                    if not isinstance(marker_type, str):
                        return [255, 0, 0, 160]
                    elif "High" in marker_type:
                        return [255, 0, 0, 160]  # Red for critical
                    elif "Medium" in marker_type:
                        return [255, 165, 0, 160]  # Orange for serious
                    else:
                        return [255, 255, 0, 160]  # Yellow for non-serious

                # IF priority column exists, assign color based on priority
                # ELSE assign default color
                csv_data["COLOR"] = (
                    csv_data["PRIORITY"].apply(get_color)
                    if "PRIORITY" in csv_data.columns
                    else [[255, 0, 0, 160]] * len(csv_data)
                )

                # Create a PyDeck map layer with circular markers
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=csv_data,
                    get_position="[LONGITUDE, LATITUDE]",
                    get_color="COLOR",
                    get_radius=1,
                    radius_min_pixels=10,
                    radius_max_pixels=10,
                    pickable=True,
                )

                # Example csv_data with latitude and longitude columns
                min_latitude = csv_data["LATITUDE"].min()
                max_latitude = csv_data["LATITUDE"].max()
                min_longitude = csv_data["LONGITUDE"].min()
                max_longitude = csv_data["LONGITUDE"].max()

                # Compute the center point (mean latitude and longitude)
                center_latitude = csv_data["LATITUDE"].mean()
                center_longitude = csv_data["LONGITUDE"].mean()

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
                    pitch=0,  # Top-down view
                )

                # Create the PyDeck map with Mapbox base style
                deck = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip={"text": "{DESCRIPTION}"},
                    map_style="mapbox://styles/mapbox/streets-v11",  # Mapbox base style
                )

                # Display the map in Streamlit
                st.pydeck_chart(deck)

        async def send_marker(session, request_url, headers, marker, i):
            delay = (
                datetime.fromisoformat(marker["event_time"]) - datetime.now()
            ).total_seconds()
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
            request_url = f"{st.session_state.api_url}/api/v0/marker"

            headers = {
                "Authorization": f"{st.session_state.api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            error_messages = []

            async with httpx.AsyncClient() as session:
                tasks = []
                for i, marker in enumerate(markers_json):
                    tasks.append(send_marker(
                        session, request_url, headers, marker, i))
                results = await asyncio.gather(*tasks)

            for result in results:
                if result:
                    error_messages.append(result)

            if error_messages:
                st.error(
                    "Errors occurred during the upload:\n\n" +
                    "\n".join(error_messages)
                )
                if any("Invalid Authorization header" in s for s in error_messages):
                    st.info(
                        "Advice: Please check your API Token and make sure it is correct."
                    )
                else:
                    st.warning(
                        "Advice: Please check the markers data and try again.")
            else:
                st.success("All markers uploaded successfully!")

        # Add a single Send Markers button at the bottom of the toolbox
        if st.button("🚨 Send Markers!"):
            asyncio.run(send_markers())


main()
