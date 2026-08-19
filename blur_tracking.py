def iou(a, b):
    ax1, ay1, aw, ah, _ = a
    bx1, by1, bw, bh, _ = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = aw * ah + bw * bh - overlap
    return overlap / union if union else 0.0


def keep_recent_boxes(boxes, held):
    return boxes + [box for box, ttl in held if ttl > 0 and all(iou(box, current) < 0.3 for current in boxes)]
