import config, os, shutil

available_classes = ["A", "C", "D", "G", "H", "M", "N", "O"]

dataset_name = str(input("Enter dataset name:\n"))
datasetPath = os.path.sep.join([".", config.DATASET_DATASET_PATH, dataset_name])
os.makedirs(datasetPath)

while True:
    category_name = str(input("Enter category name:\n"))
    catPath = os.path.sep.join([datasetPath, category_name])
    os.makedirs(catPath)

    classes = str(input("Enter classes that belong to this category (delimited by comma):\nAvailable classes : %s\n"
        %(available_classes))).split(",")
    for c in classes:
        if not c.strip() in available_classes:
            print("Invalid! No '%s' in available classes!"%(c.strip()))
            quit()
        available_classes.remove(c.strip())
        classPath = os.path.sep.join([".", config.DATASET_CLASSES_CROPPED_PATH, c.strip()])
        images = os.listdir(classPath)
        for image in images:
            imagePath = os.path.sep.join([classPath, image])
            shutil.copy2(imagePath, catPath)
    
    if available_classes == []:
        print("Bye!")
        quit()

    prompt = str(input("Build another category (y/n)?\n"))
    if prompt == "n":
        print("Bye!")
        quit()
    elif prompt != "y":
        print("Invalid!")
        quit()