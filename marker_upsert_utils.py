import random
from datetime import datetime, timedelta, timezone
import json
import requests

UPSERT_MARKER_MUTATION = """
mutation upsertIncidentLocationMarker($orgUuid: UUID!, $source: MarkerSource!, $type: MarkerType!, $json_metadata: String) {
  upsertMarker(
    input: {orgUuid: $orgUuid, source: $source, type: $type, jsonMetadata: $json_metadata}
  ) {
    upsertedMarker {
      uuid
      created
      source
      type
      status
      location {
        latitude
        longitude
        address
      }
      jsonMetadata
    }
    errors
  }
}
"""

CLEAR_MARKERS_AND_OPERATIONS_MUTATION = """
mutation clearMarkersAndOperations($orgUuid: UUID!) {
  clearOrgOperationsAndMarkers(orgUuid:$orgUuid) {
    markersDeleted
    operationsDeleted
  }
}
"""

# Bounding box around the Skydio HQ area
BOUNDING_BOX_BOTTOM_LEFT = {"latitude": 37.519775748232014, "longitude": -122.35869380303053}
BOUNDING_BOX_UPPER_RIGHT = {"latitude": 37.548770498145196, "longitude": -122.30427716250712}
MAX_DELTA_TIME_SECONDS = 3600 * 24  # 1 day
MARKER_TYPE_TO_SOURCE = {
    "INCIDENT_LOCATION_CRITICAL_PRIORITY": "INCIDENT_LOCATION",
    "INCIDENT_LOCATION_HIGH_PRIORITY": "INCIDENT_LOCATION",
    "INCIDENT_LOCATION_MEDIUM_PRIORITY": "INCIDENT_LOCATION",
}


def clear_markers_and_operations(api_url, org_uuid, api_key):
    """Clear all markers and operations."""
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    query = {
        "query": CLEAR_MARKERS_AND_OPERATIONS_MUTATION,
        "variables": {"orgUuid": org_uuid}
    }

    response = requests.post(api_url, json=query, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error clearing markers and operations: {response.text}")


def generate_markers(count):
    """Generate random incident location markers within a bounding box."""
    markers = []
    marker_types = random.choices(
        ["INCIDENT_LOCATION_CRITICAL_PRIORITY", "INCIDENT_LOCATION_HIGH_PRIORITY", "INCIDENT_LOCATION_MEDIUM_PRIORITY"],
        k=count
    )

    for index, marker_type in enumerate(marker_types):
        location = {
            "latitude": random.uniform(BOUNDING_BOX_BOTTOM_LEFT["latitude"], BOUNDING_BOX_UPPER_RIGHT["latitude"]),
            "longitude": random.uniform(BOUNDING_BOX_BOTTOM_LEFT["longitude"], BOUNDING_BOX_UPPER_RIGHT["longitude"]),
        }
        event_time = (datetime.now() - timedelta(seconds=random.randint(0, MAX_DELTA_TIME_SECONDS))).isoformat()
        
        marker = {
            "type": marker_type,
            "json_metadata": {
                "description": f"Mock marker {index}",
                "eventTime": event_time,
                **location
            }
        }
        markers.append(marker)
    
    return markers


def upsert_markers(api_url, org_uuid, api_key, markers):
    """Upsert markers to the API."""
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    created_count = 0

    for index, marker in enumerate(markers):
        marker_type = marker["type"]
        json_metadata = json.dumps(marker["json_metadata"])

        query = {
            "query": UPSERT_MARKER_MUTATION,
            "variables": {
                "orgUuid": org_uuid,
                "source": MARKER_TYPE_TO_SOURCE[marker_type],
                "type": marker_type,
                "json_metadata": json_metadata
            }
        }

        response = requests.post(api_url, json=query, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            if "errors" in result["data"]["upsertMarker"]:
                print(f"Error creating marker {index + 1}: {result['data']['upsertMarker']['errors']}")
            else:
                created_count += 1
        else:
            print(f"Error creating marker {index + 1}: {response.text}")
    
    return created_count