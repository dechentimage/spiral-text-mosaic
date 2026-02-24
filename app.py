"""
Combined Flask application for generating spiral text mosaic GIFs.

This script contains all necessary code to extract dominant colours from
an uploaded image, build an animated spiral mosaic with a dynamic
text banner, and serve the resulting GIF via a web interface.  It
eliminates the need for external modules by embedding helper
functions directly within the file.

Usage
-----
To run locally, install dependencies from ``requirements.txt`` and
execute this script with Python::

    pip install -r requirements.txt
    python app.py

The application exposes two routes:

* ``/`` – a form for uploading an image and generating a GIF.
* ``/gif/<filename>`` – displays the generated GIF and provides a
  download link.

Dependencies
------------
This module depends on Flask, Pillow, NumPy and scikit‑learn.  The
``render.yaml`` file included in this repository configures Render to
install these dependencies and start the app using Gunicorn.
"""

from __future__ import annotations

import math
import os
import uuid
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, render_template_string, send_from_directory, flash, redirect, Response

try:
    from sklearn.cluster import KMeans  # type: ignore
except Exception:
    KMeans = None  # type: ignore


# ---------------------------------------------------------------------------
# Colour extraction and text parsing helpers
# ---------------------------------------------------------------------------

def extract_dominant_colors(image: Image.Image, num_colors: int = 5) -> List[Tuple[int, int, int]]:
    """Extract a list of dominant colours from an image using k‑means clustering.

    The image is resized to 100×100 pixels to reduce computation time.  The
    pixel values are scaled to [0,1] before clustering.  The returned
    colours are unsorted and expressed as RGB integer tuples.

    Parameters
    ----------
    image:
        A PIL ``Image`` from which colours will be extracted.
    num_colors:
        Number of colours to extract.
    Returns
    -------
    List[Tuple[int, int, int]]
        The dominant colours.
    """
    if KMeans is None:
        raise ImportError(
            "scikit-learn is required for colour clustering. Please install it before using extract_dominant_colors."
        )
    im = image.convert("RGB").resize((100, 100))
    data = np.asarray(im, dtype=np.float32) / 255.0
    pixels = data.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=0)
    kmeans.fit(pixels)
    centres = kmeans.cluster_centers_
    return [tuple((c * 255).astype(int)) for c in centres]


def extract_key_word_from_filename(file_path: str) -> str:
    """Extract a significant word from a file name.

    Strips common screenshot prefixes and stopwords in German and
    English, then returns the longest remaining word.  If no valid
    candidate is found, returns an empty string.

    Parameters
    ----------
    file_path:
        Path to the image file.

    Returns
    -------
    str
        The extracted key word or an empty string.
    """
    import re
    name = os.path.splitext(os.path.basename(file_path))[0]
    name = re.sub(r"Screenshot\s+\d{4}-\d{2}-\d{2}\s+at\s+\d{2}-\d{2}-\d{2}", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[-_]", " ", name)
    words = re.findall(r"[A-Za-zÄÖÜäöüß]+", name)
    stopwords = {
        "at",
        "of",
        "the",
        "and",
        "for",
        "to",
        "a",
        "in",
        "on",
        "am",
        "um",
        "und",
        "der",
        "die",
        "das",
        "den",
        "dem",
        "des",
        "im",
        "mit",
        "von",
        "zu",
        "nach",
        "als",
        "an",
        "für",
        "bis",
        "aus",
        "bei",
        "ist",
        "sind",
    }
    candidates = [w for w in words if w.lower() not in stopwords]
    return max(candidates, key=len) if candidates else ""


# ---------------------------------------------------------------------------
# Text mask and mosaic helpers
# ---------------------------------------------------------------------------

def _find_darkest_colour(colours: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Return the darkest colour from a list using a simple luminance metric."""
    def luminance(c: Tuple[int, int, int]) -> float:
        return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    return min(colours, key=luminance)


def _render_text_mask(text: str, block_size: int) -> Tuple[Image.Image, int, int]:
    """Render uppercase text into a binary mask and scale to mosaic resolution."""
    font = ImageFont.load_default()
    text = text.upper()
    width_px, height_px = font.getsize(text)
    mask = Image.new("1", (width_px, height_px), 0)
    draw = ImageDraw.Draw(mask)
    draw.text((0, 0), text, 1, font=font)
    width_cells = width_px
    height_cells = height_px
    scaled_mask = mask.resize((width_cells * block_size, height_cells * block_size), Image.NEAREST)
    return scaled_mask, width_cells, height_cells


def create_spiral_text_mosaic_gif(
    image: Image.Image | str,
    key_word: str | None = None,
    output_path: str = "output.gif",
    *,
    block_size: int = 12,
    num_colors: int = 5,
    frame_count: int = 30,
    frame_duration: int = 100,
) -> None:
    """Generate an animated GIF featuring a spiral mosaic and an animated text bar."""
    # Load image if path provided
    if isinstance(image, str):
        base_path = image
        image = Image.open(base_path)
    else:
        base_path = getattr(image, "filename", "")
    # Determine the key word
    word = key_word or extract_key_word_from_filename(base_path) or ""
    # Extract palette colours
    colours = extract_dominant_colors(image, num_colors=num_colors)
    if not colours:
        raise ValueError("No colours could be extracted from the image.")
    darkest = _find_darkest_colour(colours)
    # Render text mask
    text_mask, text_cells_w, text_cells_h = _render_text_mask(word, block_size)
    # Determine dimensions
    swirl_cells_w = max(1, text_cells_w)
    swirl_cells_h = max(1, swirl_cells_w // 2)
    swirl_width = swirl_cells_w * block_size
    swirl_height = swirl_cells_h * block_size
    text_width_px = text_cells_w * block_size
    text_height_px = text_cells_h * block_size
    total_width = swirl_width
    total_height = swirl_height + text_height_px
    # Precompute base angles for swirl
    centre_x = (swirl_cells_w - 1) / 2.0
    centre_y = (swirl_cells_h - 1) / 2.0
    base_angles = np.zeros((swirl_cells_h, swirl_cells_w), dtype=np.float32)
    for yi in range(swirl_cells_h):
        for xi in range(swirl_cells_w):
            x0 = xi - centre_x
            y0 = yi - centre_y
            base_angles[yi, xi] = math.atan2(y0, x0)
    frames: List[Image.Image] = []
    # Generate frames
    for frame_idx in range(frame_count):
        phi = 2 * math.pi * frame_idx / frame_count
        swirl_frame = np.zeros((swirl_height, swirl_width, 3), dtype=np.uint8)
        for yi in range(swirl_cells_h):
            for xi in range(swirl_cells_w):
                angle = base_angles[yi, xi] + phi
                theta = (angle + 2 * math.pi) % (2 * math.pi)
                idx = int(theta / (2 * math.pi) * len(colours)) % len(colours)
                colour = colours[idx]
                y_start = yi * block_size
                y_end = y_start + block_size
                x_start = xi * block_size
                x_end = x_start + block_size
                swirl_frame[y_start:y_end, x_start:x_end] = colour
        frame = Image.new("RGB", (total_width, total_height), darkest)
        frame.paste(Image.fromarray(swirl_frame), (0, 0))
        # Draw text
        mask_pixels = text_mask.load()
        for ty in range(text_cells_h):
            for tx in range(text_cells_w):
                mask_pixel = mask_pixels[tx * block_size + block_size // 2, ty * block_size + block_size // 2]
                if mask_pixel:
                    idx = (tx + frame_idx) % len(colours)
                    colour = colours[idx]
                else:
                    colour = darkest
                x_start = tx * block_size
                y_start = swirl_height + ty * block_size
                for y in range(y_start, y_start + block_size):
                    for x in range(x_start, x_start + block_size):
                        frame.putpixel((x, y), colour)
        frames.append(frame)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
        disposal=2,
    )


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__)

    # -----------------------------------------------------------------------
    # Simple HTTP Basic authentication
    #
    # To prevent unauthorized access, the application requires clients to
    # authenticate using HTTP Basic Auth.  The password is provided via the
    # environment variable TOOL_PASSWORD (falling back to a default for local
    # development).  Any username is accepted as long as the password matches.

    PASSWORD = os.environ.get("TOOL_PASSWORD", "Orlando3")

    def check_auth(auth) -> bool:
        """Return True if the provided authentication is valid."""
        return bool(auth) and auth.password == PASSWORD

    def authenticate() -> Response:
        """Return a 401 response prompting for basic auth."""
        return Response(
            "Authentication required", 401,
            {"WWW-Authenticate": 'Basic realm="Login Required"'}
        )

    @app.before_request
    def require_basic_auth():
        """Ensure the client is authenticated for protected routes."""
        # Allow access to static and generated GIF files without authentication
        if request.path.startswith("/gif/") or request.path.startswith("/download/"):
            return None
        auth = request.authorization
        if not check_auth(auth):
            return authenticate()
    app.secret_key = os.environ.get("SECRET_KEY", uuid.uuid4().hex)
    upload_dir = Path(app.instance_path) / "uploads"
    gif_dir = Path(app.instance_path) / "gifs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            file = request.files.get("screenshot")
            if not file or file.filename == "":
                flash("Bitte wähle ein Bild zum Hochladen aus.")
                return redirect(request.url)
            filename = uuid.uuid4().hex + Path(file.filename).suffix
            upload_path = upload_dir / filename
            file.save(upload_path)
            gif_filename = uuid.uuid4().hex + ".gif"
            gif_path = gif_dir / gif_filename
            try:
                create_spiral_text_mosaic_gif(
                    image=str(upload_path),
                    output_path=str(gif_path),
                    block_size=12,
                    num_colors=5,
                    frame_count=30,
                    frame_duration=120,
                )
            except Exception as exc:
                flash(f"Fehler beim Erzeugen des GIFs: {exc}")
                return redirect(request.url)
            return redirect(f"/gif/{gif_filename}")
        return render_template_string(
            """
            <!doctype html>
            <html lang="de">
            <head>
              <meta charset="utf-8">
              <title>Spiral Text Mosaic Generator</title>
              <style>
                body { font-family: Arial, sans-serif; margin: 2rem; }
                .container { max-width: 600px; margin: auto; }
              </style>
            </head>
            <body>
              <div class="container">
                <h1>Spiral Text Mosaic Generator</h1>
                <form method="post" enctype="multipart/form-data">
                  <p><input type="file" name="screenshot" accept="image/*"></p>
                  <p><button type="submit">GIF erzeugen</button></p>
                </form>
                {% with messages = get_flashed_messages() %}
                  {% if messages %}
                    <ul>
                      {% for msg in messages %}<li>{{ msg }}</li>{% endfor %}
                    </ul>
                  {% endif %}
                {% endwith %}
              </div>
            </body>
            </html>
            """
        )

    @app.route("/gif/<path:filename>")
    def serve_gif(filename: str):
        gif_path = gif_dir / filename
        if not gif_path.exists():
            flash("Die angeforderte Datei wurde nicht gefunden.")
            return redirect("/")
        return render_template_string(
            """
            <!doctype html>
            <html lang="de">
            <head><meta charset="utf-8"><title>Generiertes GIF</title></head>
            <body>
              <h1>Generiertes GIF</h1>
              <p><img src="{{ url_for('download_gif', filename=filename) }}" alt="Generiertes GIF"></p>
              <p><a href="{{ url_for('download_gif', filename=filename) }}" download>GIF herunterladen</a></p>
              <p><a href="/">Neues GIF erzeugen</a></p>
            </body>
            </html>
            """,
            filename=filename,
        )

    @app.route("/download/<path:filename>")
    def download_gif(filename: str):
        return send_from_directory(gif_dir, filename, as_attachment=True)

    return app


if __name__ == "__main__":
    app = create_app()
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)