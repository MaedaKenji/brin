# Product Requirement Document (PRD)
## Flexible & Adaptive Campus POI & Review Scraper System

| Metadata | Value |
| :--- | :--- |
| **Project Name** | Multi-Location Campus POI & Review Data Pipeline |
| **Author** | Agus Fuad Mudhofar |
| **Status** | Draft (Updated with Dynamic Location Feature) |
| **Date** | 16 Juni 2026 |

---

## 1. Executive Summary & Objective
Tujuan dari proyek ini adalah membangun sistem pengumpulan data poin penting (*Point of Interest* / POI) beserta data ulasan (*review*) yang **fleksibel, dinamis, dan fully reproducible**. 

Sistem ini tidak lagi terikat (*hardcoded*) pada satu lokasi kampus saja (seperti NCU). Sistem diperbarui agar dapat memetakan lokasi mana pun di belahan dunia dengan dua mode operasional:
1. **Mode Manual/Reproducible:** Pengguna memasukkan koordinat atau nama wilayah baru.
2. **Mode Otomatis (Auto-Scrape):** Sistem mendeteksi lokasi terkini pengguna secara *real-time*, lalu otomatis mengikis data POI dan ulasan Google Maps di area sekitar tersebut.

---

## 2. System Architecture & Workflow (Dynamic Location)

Sistem beradaptasi secara dinamis berdasarkan input lokasi yang diterima sebelum mengeksekusi pipeline pengikisan data:

```text
               +-----------------------------------------+
               | Pilihan Mode Penentuan Lokasi Pengguna |
               +--------------------+--------------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v (Mode Otomatis)                               v (Mode Manual)
+-------------------------------+               +-------------------------------+
|   Fitur: Get Current Location |               |     Input Koordinat / Area    |
|   - Geolocation API / IP-Base |               |     di File Konfigurasi       |
+-----------+-------------------+               +-----------+-------------------+
            |                                               |
            +-----------------------+-----------------------+
                                    |
                                    v Menghasilkan Pusat Koordinat (Lat, Long)
+-----------------------------------+-----------------------------------+
|                      Fase 1: Geolocation Seeding                      |
|  - Konversi Titik Koordinat menjadi Bounding Box Dinamis              |
|  - Query Overpass API (OSM) sekeliling radius target                  |
+-----------------------------------+-----------------------------------+
                                    |
                                    v Output: `poi_seed.csv`
+-----------------------------------+-----------------------------------+
|                     Fase 2: Enrichment (Selenium)                     |
|  - Buka Google Maps via Koordinat Presisi dari OSM                    |
|  - Otomatisasi Infinite Scroll & Ekstraksi Review Historis            |
+-----------------------------------------------------------------------+