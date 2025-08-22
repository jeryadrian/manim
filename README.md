# Manim Animation Project

This project contains scripts and resources for generating animations using [Manim](https://www.manim.community/), a mathematical animation engine.

## Project Structure

- `animation.py` — Main Python script for generating animations.
- `input/` — Contains input data files (e.g., `.gpkg` files) used by the animation scripts.
- `media/` — Output media files:
  - `images/` — Generated images.
  - `videos/` — Generated videos, organized by animation type and resolution.
- `output/` — Output data files (e.g., results of computations).

## Requirements

- Python 3.11 or later
- [Manim](https://www.manim.community/) (Community Edition)
- Other dependencies as required by `animation.py` (see code for details)

## Setup

1. (Recommended) Create a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install Manim and other dependencies:

   ```powershell
   pip install manim
   # Install other packages as needed
   ```

## Usage

To generate animations, run:

```powershell
python animation.py
```

Output videos and images will be saved in the `media/` directory.

## Notes

- Input files in `input/` must be present and correctly formatted for the scripts to work.
- Output files are organized by animation type and resolution.

## License

Specify your license here (e.g., MIT, GPL, etc.).
