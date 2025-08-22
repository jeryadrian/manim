
# Logistics Optimization Animation with Manim

This project visualizes the process of optimizing Temporary Storage Site (TSS) locations for logistics using a greedy algorithm. The animation, built with [Manim](https://www.manim.community/), demonstrates how candidate sites are selected to minimize transport effort between material origins and destinations (CCHs) over a real road network. The script uses geospatial data and network routing to show each step of the selection process, including:

- Drawing the road network and base map
- Displaying candidate pool points, material origins, and destinations
- Animating the greedy selection of TSS locations
- Showing how optimal routes change as new TSS sites are added
- Summarizing the final optimized network

This is useful for presentations or analysis of logistics optimization strategies in spatial planning or disaster response.

## Project Structure

- `animation.py` — Main script for generating the logistics optimization animation
- `input/` — Geospatial input data (`road.gpkg`, `grid.gpkg`, `project.gpkg`, `cch.gpkg`)
- `output/` — Output data (e.g., `optimal_tss_locations.gpkg` with selected TSS sites)
- `media/` — Generated images and videos

## Requirements

- Python 3.11 or later
- [Manim](https://www.manim.community/) (Community Edition)
- geopandas, pandas, numpy, shapely, igraph, scipy

## Setup

1. (Recommended) Create a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install Manim and required dependencies:

   ```powershell
   pip install manim geopandas pandas numpy shapely python-igraph scipy
   ```

## Usage

1. Prepare the required input files in the `input/` and `output/` folders:
   - `input/road.gpkg` — Road network
   - `input/grid.gpkg` — Candidate pool points
   - `input/project.gpkg` — Material origins
   - `input/cch.gpkg` — Destinations (CCH)
   - `output/optimal_tss_locations.gpkg` — Precomputed optimal TSS locations

2. To preview the animation (for development):

   ```powershell
   python animation.py
   ```

   This will not render a video, but will check that the script runs and data loads.

3. To render the animation as a video (recommended):

   ```powershell
   manim -pql animation.py LogisticsOptimizationAnimation
   ```

   The output video will be saved in the `media/videos/animation/` directory.

## Notes

- Input files must be present and correctly formatted for the animation to work.
- The animation visualizes the greedy selection of TSS locations and the resulting changes in optimal routes.
- Output files are organized by animation type and resolution.

## License

Specify your license here (e.g., MIT, GPL, etc.).
