import cv2
import numpy as np
import tensorflow as tf


def read_image(path):
    image0 = cv2.imread(path)
    image = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
    image, padding = letterbox(image, (640, 640))
    image = (image / 255.0).astype(np.float32)
    return image0, image, padding


def letterbox(image, new_size):
    """Resize and pad the image.
    Args:
        image: input image (numpy.ndarray)
        new_size: output image shape (height, width)
    """
    size = image.shape[:2]
    f = min(new_size[0] / size[0], new_size[1] / size[1])
    h, w = int(round(size[0] * f)), int(round(size[1] * f))
    dh, dw = new_size[0] - h, new_size[1] - w
    top, bottom = int(dh/2), dh - int(dh/2)
    left, right = int(dw/2), dw - int(dw/2)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=(255,255,255))
    return image, (top, bottom, left, right)


def xywh2yxyx(xywh):
    """(xc, yc, w, h) to (y1, x1, y2, x2)"""
    boxes = np.zeros_like(xywh)
    boxes[:, 0] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 1] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 2] = xywh[:, 1] + xywh[:, 3] / 2
    boxes[:, 3] = xywh[:, 0] + xywh[:, 2] / 2
    return boxes


def rescale_boxes(boxes, image0, image, padding):
    size = image.shape[:2]
    size0 = image0.shape[:2]
    top, bottom, left, right = padding
    w, h = size[1] - left - right, size[0] - top - bottom
    boxes[... , :] *= [size[0], size[1], size[0], size[1]]
    boxes[... , :] -= [top, left, top, left]
    boxes[... , :] /= [h, w, h, w]
    boxes[... , :] *= [size0[0], size0[1], size0[0], size0[1]]
    return boxes.astype(np.int32)


def non_max_suppression(prediction, score_thres, iou_thresh):
    scores = prediction[..., 4]
    prediction = prediction[scores > score_thres]

    boxes = prediction[:,:4]
    boxes = xywh2yxyx(boxes)
    scores = prediction[..., 4]

    selected_indices = tf.image.non_max_suppression(
            boxes=boxes, scores=scores, max_output_size=30,
            iou_threshold=iou_thresh)
    selected_boxes = tf.gather(boxes, selected_indices)
    return selected_boxes.numpy()


def draw_boxes(image, boxes, color):
    image = image.copy()
    for box in boxes:
        cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), color, 3)
    return image


if __name__ == '__main__':
    
    image0, image, padding = read_image('images/test01.jpg')
    model = tf.saved_model.load('lymph_saved_model')
    prediction = model(image[None]).numpy()

    boxes = non_max_suppression(prediction[0],
                                score_thres=0.25,
                                iou_thresh=0.3)

    boxes = rescale_boxes(boxes, image0, image, padding)

    image = draw_boxes(image0, boxes, (0, 255, 0))

    cv2.imwrite('result.jpg', image)
