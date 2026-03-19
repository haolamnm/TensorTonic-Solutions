def generate_boxes(x, y, scales, ratios):
    boxes = []
    for s in scales:
        for r in ratios:
            w = s * (r**0.5)
            h = s / (r**0.5)
            box = [
                x - w / 2,
                y - h / 2,
                x + w / 2,
                y + h / 2
            ]
            boxes.append(box)
    return boxes

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """

    stride = image_size / feature_size
    anchors = []

    for i in range(feature_size):
        for j in range(feature_size):
            box = []
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            boxes = generate_boxes(cx, cy, scales, aspect_ratios)
            for box in boxes:
                anchors.append(box)
            
    return anchors