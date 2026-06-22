# PRD — Building POI Direction Awareness System for NCU Area

## 1. Product Overview

The system will identify **buildings / POIs around the NCU area** and determine their **relative direction** from any given input latitude and longitude.

Example use case:

Input:

* latitude = `...`
* longitude = `...`

Output:

```json
{
  "north": "Engineering Building",
  "south": "Library",
  "east": "Cafeteria",
  "west": "Parking Hall"
}
```

This enables location intelligence for:

* navigation systems
* UAV / robotics perception
* map-based situational awareness
* smart campus applications

---

## 2. Problem Statement

Given an arbitrary GPS coordinate inside or near NCU, the system currently does not know:

* which building is nearby
* in which direction each building lies
* how to convert raw lat/lng into human-readable spatial context

The product solves this by mapping coordinates into:

* nearest POIs
* directional sectors
* distance-based relevance

---

## 3. Objectives

### Primary Objective

Build a service that returns **POIs grouped by compass direction** relative to an input coordinate.

### Secondary Objectives

* scalable for all campus buildings
* fast real-time response (< 500 ms)
* reusable as API for other applications

---

## 4. Scope

### In Scope

* POI extraction for all buildings in NCU
* coordinate input API
* direction calculation
* nearest building detection
* radius-based filtering

### Out of Scope (Phase 1)

* indoor room-level navigation
* floor-level localization
* real-time moving object tracking

---

## 5. Functional Requirements

### FR-1 POI Database

The system must store all building POIs:

* building name
* centroid latitude
* centroid longitude
* polygon footprint (optional)

Example schema:

```json
{
  "id": 1,
  "name": "Library",
  "lat": 24.9681,
  "lng": 121.1912
}
```

---

### FR-2 Input Coordinate

User / system provides:

```json
{
  "lat": ...,
  "lng": ...
}
```

---

### FR-3 Nearby Building Search

System must return all buildings within configurable radius.

Example:

* 100m
* 250m
* 500m

---

### FR-4 Direction Classification

Each building must be assigned direction:

* North
* North-East
* East
* South-East
* South
* South-West
* West
* North-West

---

### FR-5 Sorted by Distance

Buildings in each direction must be sorted nearest first.

---

## 6. Technical Architecture

```text
POI Data Source
     ↓
Data Preprocessing
     ↓
Spatial Database
     ↓
Direction Engine
     ↓
REST API
     ↓
Application / Frontend
```

---

## 7. Step-by-Step Development Plan

# Phase 1 — Data Collection

## Step 1. Define NCU Area Boundary

Define campus bounding box:

```text
min_lat
max_lat
min_lng
max_lng
```

This will be used for POI extraction.

---

## Step 2. Collect Building POIs

Use OpenStreetMap / campus map source.

Collect:

* building names
* polygons
* centroid points

Output file:

```text
poi_buildings.csv
```

Columns:

```text
id, name, lat, lng
```

---

## Step 3. Clean POI Data

Normalize:

* missing names
* duplicate buildings
* unnamed polygons

Example:

```text
Building A
Engineering Hall
Library
Lab 1
```

---

# Phase 2 — Spatial Engine

## Step 4. Build Distance Calculator

Implement Haversine distance.

Purpose:

* find nearest buildings
* filter by radius

---

## Step 5. Build Bearing Calculator

Calculate azimuth from input point to POI.

This gives angle:

```text
0°   = North
90°  = East
180° = South
270° = West
```

---

## Step 6. Direction Sector Mapping

Convert angle into sectors:

```text
337.5–22.5   → North
22.5–67.5    → North-East
67.5–112.5   → East
...
```

---

# Phase 3 — API Development

## Step 7. Build Query API

Example:

```http
POST /direction-poi
```

Input:

```json
{
  "lat": 24.968,
  "lng": 121.191
}
```

Output:

```json
{
  "north": ["Library"],
  "south": ["Cafeteria"],
  "east": ["Engineering Hall"],
  "west": []
}
```

---

## Step 8. Radius Parameter

Allow configurable search:

```json
{
  "lat": ...,
  "lng": ...,
  "radius": 200
}
```

---

# Phase 4 — Optimization

## Step 9. Fast Spatial Search

Use:

* KDTree
* BallTree
* PostGIS spatial index

This is important if building count increases.

---

## Step 10. Testing

Test scenarios:

### Test A

Input exactly center of campus

### Test B

Input near building edge

### Test C

Input outside campus

Expected:

* no crash
* empty list if no POI

---

## 8. Success Metrics

* response time < 500 ms
* direction accuracy > 95%
* correct nearest POI ranking

---

## 9. Risks

### Risk 1

POI map data incomplete

Mitigation:

* manual campus validation

### Risk 2

GPS noise

Mitigation:

* use 5–10 meter tolerance

---

## 10. Future Roadmap

Phase 2 features:

* real-time map visualization
* arrow-based navigation
* route recommendation
* voice direction output
