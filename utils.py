import io
from pathlib import Path
from uuid import uuid4

from PIL import Image
from rembg import remove


NEUTRAL_GRAY = (230, 230, 230, 255)


def process_and_save_uploaded_image(uploaded_file, uploads_dir: Path) -> str:
    """
    Removes image background and places the item on a neutral gray canvas.
    Returns absolute path to the saved PNG.
    """
    uploads_dir.mkdir(parents=True, exist_ok=True)
    input_bytes = uploaded_file.getvalue()
    no_bg_bytes = remove(input_bytes)

    foreground = Image.open(io.BytesIO(no_bg_bytes)).convert("RGBA")
    gray_background = Image.new("RGBA", foreground.size, NEUTRAL_GRAY)
    composed = Image.alpha_composite(gray_background, foreground)

    filename = f"{uuid4().hex}.png"
    output_path = uploads_dir / filename
    composed.convert("RGB").save(output_path, format="PNG", optimize=True)
    return str(output_path)
