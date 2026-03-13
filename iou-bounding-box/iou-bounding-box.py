from itertools import chain

def area(x1, y1, x2, y2):
    x = x2 - x1
    y = y2 - y1
    return x * y


def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    xa1, ya1, xa2, ya2 = tuple(box_a)
    xb1, yb1, xb2, yb2 = tuple(box_b)

    # Intersection
    xi1, yi1 = max((xa1, ya1), (xb1, yb1))
    xi2, yi2 = min((xa2, ya2), (xb2, yb2))

    if xi1 >= xi2 and yi1 >= yi2:
        return 0.0
    
    area_i = area(xi1, yi1, xi2, yi2)

    # Union
    area_a = area(xa1, ya1, xa2, ya2)
    area_b = area(xb1, yb1, xb2, yb2)
    area_u = area_a + area_b - area_i

    return area_i  / area_u
    