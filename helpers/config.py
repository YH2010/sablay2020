import os, time

TIME = time.strftime("%m%d%y%H%M%S", time.localtime(time.time()))

ORIGINAL_INPUT_TRAINING_IMAGES_PATH = "ODIR-5K/ODIR-5K_Training_Images"
ORIGINAL_INPUT_TESTING_IMAGES_PATH = "ODIR-5K/ODIR-5K_Testing_Images"
ORIGINAL_INPUT_LABELS_FILE = "ODIR-5K/ODIR-5K_Training_Annotations(Updated)_V2.csv"

DATASET_CLASSES_PATH = "dataset/classes"
DATASET_DATASET_PATH = "dataset/dataset"
OUTPUT_PATH = "output"

TRAIN = "training"
TEST = "testing"