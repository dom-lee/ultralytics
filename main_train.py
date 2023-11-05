import argparse

from ultralytics import YOLO

def get_parser():
    """
    Create and return an argument parser for command line arguments.
    """
    parser = argparse.ArgumentParser(
            description='Train YOLO model with specified parameters.')
    parser.add_argument('--model',
                        choices=['yolov8n.pt', 'yolov8n-oiv7.pt',
                                 'yolov8s.pt', 'yolov8s-oiv7.pt',
                                 'yolov8m.pt', 'yolov8m-oiv7.pt',
                                 'yolov8l.pt', 'yolov8l-oiv7.pt',
                                 'yolov8x.pt', 'yolov8x-oiv7.pt'],
                        default='yolov8n.pt',
                        help="pretrained model")
    parser.add_argument('--data', type=str,
                        default='/home/dongmyeong/Projects/AMRL/CAO-SLAM/detector/ultralytics/datasets/CODa/CODa.yaml', 
                        help='Path to the dataset configuration file.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for.')
    parser.add_argument('--imgsz', type=int, default=1224,
                        help='Image size for training.')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)

