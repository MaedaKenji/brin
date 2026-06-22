# Surrounding Awareness API — Postman Guide

This guide provides instructions on how to test and use the Surrounding Awareness API using [Postman](https://www.postman.com/).

## 1. Prerequisites

- **Postman** installed on your machine.
- The API server running. Start it with:
  ```powershell
  python final.py
  # OR
  uvicorn final:app --reload --port 8000
  ```
- **Base URL**: `http://localhost:8000`

---

## 2. API Endpoints

### 🩺 Health Check
**GET** `http://localhost:8000/api/health`

Used to verify if the server is up and the model is loaded.

### 🔍 Analyze Image
**POST** `http://localhost:8000/api/analyze`

The primary endpoint for object detection, action recognition, and GPS-based POI lookup.

---

## 3. Step-by-Step: Analyzing an Image

Follow these steps in Postman to send a request to the `/api/analyze` endpoint:

1.  **Create a New Request**:
    -   Set method to **POST**.
    -   Enter URL: `http://localhost:8000/api/analyze`.

2.  **Set Body Type**:
    -   Go to the **Body** tab.
    -   Select **form-data**.

3.  **Configure Parameters**:
    Add the following keys in the form-data table:

    | Key | Type | Value | Description |
    | :--- | :--- | :--- | :--- |
    | `image` | **File** | `[Select your image]` | Click "Select Files" to upload a JPEG or PNG. |
    | `lat` | Text | `24.968` | (Optional) Latitude. If omitted, the API tries to extract EXIF data. |
    | `lng` | Text | `121.191` | (Optional) Longitude. |
    | `gps_radius` | Text | `500` | (Optional) POI search radius in meters (Default: 500). |

    > [!TIP]
    > To change a key's type to **File** in Postman, hover over the key row, click the dropdown that appears on the right of the "Key" field, and select **File**.

4.  **Send the Request**:
    -   Click the **Send** button.

---

## 4. Understanding the Response

The API returns a JSON object with the following sections:

- **`objects`**: Detected items (car, person, etc.) with their counts, confidence, and dominant colors.
- **`actions`**: Detected activities (walking, cycling, etc.) and their counts.
- **`doors`**: List of detected doors with their relative direction (`left`, `front`, or `right`).
- **`summary`**: A simplified count of all detected objects and actions.
- **`gps`**: Information about the resolved location and nearby POIs (if GPS data was provided or found in EXIF).

### Example Response Snippet
```json
{
  "objects": {
    "car": {
      "count": 1,
      "instances": [
        { "id": 0, "confidence": 0.92, "colors": ["#3a3a3a", "#ffffff"] }
      ]
    }
  },
  "actions": {
    "walking": { "count": 2 }
  },
  "doors": [
    {
      "id": 5,
      "direction": "front",
      "description": "There is a door on your front"
    }
  ],
  "gps": {
    "source": "user_input",
    "lat": 24.968,
    "lng": 121.191,
    "poi_result": {
      "nearest": { "name": "Library", "distance_m": 45.2, "direction": "North" }
    }
  }
}
```

---

## 5. Import Collection (Fast Way)

You can copy the JSON below, save it as `surrounding_awareness.postman_collection.json`, and import it directly into Postman.

<details>
<summary>Click to expand Postman Collection JSON</summary>

```json
{
	"info": {
		"_postman_id": "7a8b9c0d-1e2f-4a3b-8c9d-0e1f2a3b4c5d",
		"name": "Surrounding Awareness API",
		"schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
	},
	"item": [
		{
			"name": "Health Check",
			"request": {
				"method": "GET",
				"header": [],
				"url": {
					"raw": "http://localhost:8000/api/health",
					"protocol": "http",
					"host": [ "localhost" ],
					"port": "8000",
					"path": [ "api", "health" ]
				}
			},
			"response": []
		},
		{
			"name": "Analyze Image",
			"request": {
				"method": "POST",
				"header": [],
				"body": {
					"mode": "formdata",
					"formdata": [
						{ "key": "image", "type": "file", "src": [] },
						{ "key": "lat", "value": "24.968", "type": "text", "disabled": true },
						{ "key": "lng", "value": "121.191", "type": "text", "disabled": true },
						{ "key": "gps_radius", "value": "500", "type": "text", "disabled": true }
					]
				},
				"url": {
					"raw": "http://localhost:8000/api/analyze",
					"protocol": "http",
					"host": [ "localhost" ],
					"port": "8000",
					"path": [ "api", "analyze" ]
				}
			},
			"response": []
		}
	]
}
```
</details>
