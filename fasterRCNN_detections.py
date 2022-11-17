
"""Faster R-CNN

Pretrained Faster RCNN on Open Images V4 Dataset with 600 categories.
Object detection model trained on Open Images V4 with ImageNet pre-trained Inception Resnet V2 as image feature extractor.

Categories of interest from sun rgbd | possible category of Open Images Dataset
- bed | Bed
- table | Table
- sofa | Sofa bed
- chair | Chair
- toilet | Toilet
- desk | Desk
- dresser | Filing cabinet
- night_stand | Nightstand
- bookshelf | Bookcase
- bathtub | Bathtub
"""

import os
# For running inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For filtering out objects of interest from 600 categories
from collections import defaultdict

# For measuring the inference time.
import time

# Print Tensorflow version
print(tf.__version__)

# Check available GPU devices.
print(tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def display_image(image):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(False)
    plt.imshow(image)


def download_and_resize_image(url, new_width=256, new_height=256, display=False):
    """
    To test on sample images from the internet
    """
    _, filename = tempfile.mkstemp(suffix=".jpg")
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    print("Image downloaded to %s." % filename)
    if display:
        display_image(pil_image)
    return filename


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()):
    """
    Adds a bounding box to an image.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    # since bbox coordinates are normalized between 0 to 1
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height

    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.15):
    """
    Overlay labeled boxes on an image with formatted scores and label names.
    """
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

# Main Function which uses pretrained detector from tensorflow hub and inputs the image path and dump directory file path
def run_detector(detector, path, filePath):
    img = load_img(path)

    image = img.numpy()
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    im_width, im_height = image.size

    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key: value.numpy() for key, value in result.items()}

    # new dictionary which filters out objects of interest as mentioned at the start
    filter = defaultdict(list)
    for i in range(len(result["detection_class_entities"])):
        if result["detection_class_entities"][i] == b"Bed" or \
           result["detection_class_entities"][i] == b"Kitchen & dining room table" or \
           result["detection_class_entities"][i] == b"Table" or \
           result["detection_class_entities"][i] == b"Sofa bed" or \
           result["detection_class_entities"][i] == b"Chair" or \
           result["detection_class_entities"][i] == b"Toilet" or \
           result["detection_class_entities"][i] == b"Filing cabinet" or \
           result["detection_class_entities"][i] == b"Desk" or \
           result["detection_class_entities"][i] == b"Nightstand" or \
           result["detection_class_entities"][i] == b"Bookcase" or \
           result["detection_class_entities"][i] == b"Bathtub":

            filter["detection_class_entities"].append(result["detection_class_entities"][i])
            filter["detection_boxes"].append(result["detection_boxes"][i])
            filter["detection_scores"].append(result["detection_scores"][i])

    # print(filter["detection_class_entities"])
    # print(filter["detection_boxes"])
    # print(filter["detection_scores"])

    print("Found %d objects." % len(result["detection_scores"]))
    print("Inference time: ", end_time - start_time)

    # code to save all detected objects in a local text file (as per ImVoteNet requirements)
    currentFile = open(filePath, mode='w')
    for i in range(len(filter["detection_class_entities"])):
        xmin = filter["detection_boxes"][i][0] * im_width
        xmax = filter["detection_boxes"][i][2] * im_width
        ymin = filter["detection_boxes"][i][1] * im_height
        ymax = filter["detection_boxes"][i][3] * im_height
        if str(filter["detection_class_entities"][i].decode("ascii")) == 'Bed':
            className = 'bed'
        elif str(filter["detection_class_entities"][i].decode("ascii")) == 'Table':
            className = 'table'
        elif str(filter["detection_class_entities"][i].decode("ascii")) == 'Sofa bed':
            className = 'sofa'
        elif str(filter["detection_class_entities"][i].decode("ascii")) == 'Chair':
            className = 'chair'
        elif str(filter["detection_class_entities"][i].decode("ascii")) == 'Toilet':
            className = 'toilet'
        elif str(filter["detection_class_entities"][i].decode("ascii")) == 'Desk':
            className = 'desk'
        elif str(filter["detection_class_entities"][i].decode("ascii")) == 'Filing Cabinet':
            className = 'dresser'
        elif str(filter["detection_class_entities"][i].decode("ascii")) == 'Nightstand':
            className = 'night_stand'
        elif str(filter["detection_class_entities"][i].decode("ascii")) == 'Bookcase':
            className = 'bookshelf'
        elif str(filter["detection_class_entities"][i].decode("ascii")) == 'Bathtub':
            className = 'bathtub'
        currentFile.write(
            className + ' ' + '0' + ' ' + '0' + ' ' + '-10' + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(
                xmax) + ' ' + str(ymax) + ' ' + str(filter["detection_scores"][i]) + '\n')
    currentFile.close()

    image_with_boxes = draw_boxes(img.numpy(), np.array(filter["detection_boxes"]),
                                  np.array(filter["detection_class_entities"]), np.array(filter["detection_scores"]))
    display_image(image_with_boxes)


module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
img_path = os.path.join(BASE_DIR, 'demo/image/000001.jpg')
# path at which resulting text file needs to be dumped
filePath = os.path.join(BASE_DIR, 'demo/FasterRCNN_labels/textfile.txt')
run_detector(detector, img_path, filePath)