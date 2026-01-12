# Star Localizator

Star Localizator is a student project for the course **Statistical Methods in AI for Big Data Analyses**.

The goal of the project is to build a pipeline that:
1. Generates a labeled astronomical image dataset using star catalogs and survey data.
2. Trains a Convolutional Neural Network (CNN) to detect stars in sky images.
3. Uses detected star patterns for identification and localization.


## Data sources

### Star catalog
- **HYG Database v4.1**
- Source: https://github.com/astronexus/HYG-Database
- Used fields:
  - `id` – internal star identifier
  - `ra`, `dec` – sky coordinates (RA in hours, Dec in degrees)
  - `mag` – apparent magnitude

### Astronomical images
- **SDSS (Sloan Digital Sky Survey)**
- Accessed via `astroquery.sdss`
- FITS images in the `r` photometric band

---

## Current functionality (implemented)

### Dataset generation
The project generates a dataset suitable for CNN training by:

1. Reading stars from the HYG catalog.
2. Querying SDSS for images near each star position.
3. Downloading FITS images from SDSS.
4. Converting sky coordinates (RA/Dec) to pixel coordinates using WCS.
5. Extracting fixed-size image patches centered on stars.
6. Saving:
   - image data as NumPy arrays (`.npy`)
   - labels as JSON files with star metadata and pixel coordinates
7. Creating a deterministic train/validation split.

---

## Data Source

- https://github.com/astronexus/HYG-Database/blob/main/hyg/CURRENT/hygdata_v41.csv

- zczytujemy stąd dane gwiazd (nazwa, koordynaty, id)

- używając query (przykład w main) szukamy zdjęć dla każdej gwiazdy i zapisujemy je w bazie

- forma zapisu danych:
    - zdjecie -> lista (id, koordynaty na zdj, wielkość na zdj)

- CNN - wykrywa gwiazdy na zdj i próbuje je rozpoznać

