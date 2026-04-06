import numpy as np

def nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression.
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    # coordinates of boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    indices = np.arange(len(scores))
    order = np.lexsort((indices, -np.array(scores)))

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # find the coordinates of the intersection rectangle
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # compute width and height of intersection
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # missed this
        inds = np.where(ovr < iou_threshold)[0]
        
        # update 'order' to only include the remaining boxes
        # we add 1 because ovr was calculated on order[1:]
        order = order[inds + 1]

    return np.array(keep).tolist()