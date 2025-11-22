import base64
import io
from PIL import Image, ImageDraw
from .disease_names import COLORS, DISEASE_NAMES


def draw_boxes(img, boxes, scores, classes):
    draw = ImageDraw.Draw(img)

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(float, box)
        color = COLORS.get(int(cls), "#ff0000")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = f"{DISEASE_NAMES.get(int(cls))} {score*100:.1f}%"
        draw.text((x1, y1), label, fill=color)

    # convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return b64
