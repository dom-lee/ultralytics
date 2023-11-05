import os
import argparse
import cv2
from PIL import Image
from tqdm import tqdm 

from ultralytics import YOLO
from ultralytics.utils import yaml_load
from ultralytics.utils.plotting import Colors

def get_parser():
    """
    Create and return an argument parser for command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Train YOLO model with specified parameters.'
    )
    parser.add_argument('--model', type=str,
                        default='runs/detect/train12/weights/best.pt',
                        help="Path to the model file")
    parser.add_argument('--data', type=str,
                        default='datasets/CODa/CODa.yaml',
                        help="Path to the dataset configuration file")
    parser.add_argument('--images_dir', type=str,
                        default='../../datasets/CODa/images',
                        help="Path to image folder for object detection")
    parser.add_argument('--output', type=str,
                        default='../../datasets/CODa/object_detections',
                        help="Path to output folder for object detection")
    parser.add_argument('--visualize', action='store_true',
                        help="Visualize the results")
    return parser


def run_detection(args):
    """
    Run object detection on images in the dataset folder
    """
    # Load model
    model = YOLO(args.model)
    classes = yaml_load(args.data)['names']
    color_palette = Colors()
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    # Get image files
    image_files = [os.path.join(args.images_dir, file)
                   for file in os.listdir(args.images_dir)
                   if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Run inference on images
    for image_file in tqdm(image_files):
        results = model(image_file, verbose=False)
        image = cv2.imread(image_file)
        bboxes = results[0].boxes.cpu().numpy()

        # Save results
        image_name = os.path.basename(image_file).split('.')[0]
        output_file = os.path.join(args.output, f'{image_name}.txt')
        with open(output_file, 'w') as f:
            for (box, conf, cls) in zip(bboxes.xyxy, bboxes.conf, bboxes.cls):
                x1, y1, x2, y2 = map(int, box)
                f.write(f'{int(cls)} {conf:.5f} {x1} {y1} {x2} {y2}\n')

        if not args.visualize:
            continue

        # Visualization
        for (box, conf, cls) in zip(bboxes.xyxy, bboxes.conf, bboxes.cls):
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                          color_palette(int(cls), bgr=True), 1, cv2.LINE_AA)
            cv2.putText(image, f'{classes[cls]}: {conf:.3f}',
                        (int(x1), int(y1 - 9)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color_palette(int(cls), bgr=True), 2, cv2.LINE_AA)
        cv2.imshow('demo', image)
        cv2.waitKey(0)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run_detection(args)
