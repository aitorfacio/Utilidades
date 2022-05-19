import cv2
import numpy as np
from imutils import paths
from pathlib import Path
from argparse import ArgumentParser


#initialize a color list of twenty colors
colors = np.random.uniform(0, 255, size=(20, 3))


#read an image and its annotations and use cv2 to overlay a bounding box in a copy of the image
def overlay_bounding_boxes(image, boxes):
    # Make a copy of the image
    imcopy = np.copy(image)
    # Iterate through the bounding boxes
    for bbox in boxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[1], bbox[2], colors[bbox[0]], 2)
        # Draw a label given bbox coordinates, if the bounding box upper corner is too close to the image edge
        # draw it beneath it
        text_corner = (bbox[1][0] + 5, bbox[1][1])
        if bbox[1][1] < 20:
            text_corner = (text_corner[0], text_corner[1] + 20)
        cv2.putText(imcopy, bbox[0], text_corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[bbox[0]], 1)
    # Return the image copy with boxes drawn
    return imcopy


#read the annotations and return a list of bounding boxes converted from YOLO format to image coordinates
def convert_yolo_to_image_coordinates(boxes, image_shape, classes):
    # Define a bounding box
    bbox = []
    image_height, image_width = image_shape
    # Iterate through the bounding boxes
    for box in boxes:
        # Get the center x, y coordinates, width and height of the bounding box
        class_id , x, y, width, height = [float(x) for x in box]
        class_name = classes[int(class_id)]

        # Calculate the top left corner of the bounding box
        xmin = int((x - width / 2) * image_width)
        ymin = int((y - height / 2) * image_height)
        # Calculate the bottom right corner of the bounding box
        xmax = int((x + width / 2) * image_width)
        ymax = int((y + height / 2) * image_height)
        # Append the bounding box to the list
        bbox.append((class_name, (xmin, ymin), (xmax, ymax)))
    return bbox


# apply the overlay function to all images in a folder
def apply_overlay(folder, output_folder, classes):
    # list all images in the folder
    images = [Path(x) for x in paths.list_images(folder)]
    # iterate through the images
    for image_path in images:
        # read the image
        image = cv2.imread(str(image_path))
        # read the image annotations
        annotations = []
        try:
            with open(image_path.with_suffix('.txt')) as f:
                content = f.readlines()
                # Convert the content of the annotations into a list of bounding boxes
                annotations = [line.split() for line in content]
        except FileNotFoundError:
            #if there is no annotation file, skip the image
            continue
        # convert the annotations to bounding boxes
        boxes = convert_yolo_to_image_coordinates(annotations, image.shape[:2], classes)
        # apply the overlay function
        overlay = overlay_bounding_boxes(image, boxes)
        # save the image
        # add a suffix to the image name to indicate that it has been processed
        cv2.imwrite(str(output_folder / Path(image_path.stem + '_overlay.jpg')), overlay)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-i', '--input', required=True, help='Path to the input folder')
    args.add_argument('-o', '--output', required=True, help='Path to the output folder')
    args.add_argument('-c', '--classes', required=True, help='Path to the classes file')
    args = args.parse_args()
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
        # assign a color to each class in a dictionary
        colors = {classes[i]: colors[i] for i in range(len(classes))}

    apply_overlay(args.input, args.output, classes)