"""Generate visually distinct colors for track IDs."""
import colorsys


# Pre-defined high-contrast palette (first 20 tracks)
PALETTE = [
    (255, 0, 0),     (0, 255, 0),     (0, 100, 255),   (255, 255, 0),
    (255, 0, 255),   (0, 255, 255),   (255, 128, 0),   (128, 0, 255),
    (0, 255, 128),   (255, 0, 128),   (128, 255, 0),   (0, 128, 255),
    (255, 128, 128), (128, 255, 128), (128, 128, 255), (255, 200, 0),
    (200, 0, 255),   (0, 200, 128),   (200, 128, 0),   (0, 128, 200),
]


def generate_unique_color(track_id: int) -> tuple[int, int, int]:
    """
    Return a unique, visually distinct RGB color for a given track ID.
    Uses palette for first 20, then generates via golden ratio hue spacing.
    """
    if track_id <= len(PALETTE):
        return PALETTE[track_id - 1]

    # Golden ratio hue spacing for unlimited distinct colors
    golden = 0.618033988749895
    hue = (track_id * golden) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))
