import os
import xml.etree.ElementTree as ET
import PIL

# Order for bbox: xmin, ymin, xmax, ymax


annotation_path = "FaceMaskDataset/FaceMaskAnnotations"
image_path = "FaceMaskDataset/FaceMaskImages"

dataset_root = r"C:\Users\Adel\PycharmProjects\yolov8\datasets"
image_root = "images"
label_root = "labels"

label_lookup = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}


def load_data():
    annotations = []
    images = []

    for file in os.listdir(annotation_path):
        if file.endswith(".xml"):
            tree = ET.parse(os.path.join(annotation_path, file))
            root = tree.getroot()

            labels = []
            bboxes = []

            # Extract data
            for obj in root.findall('object'):
                name = obj.find('name').text
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)
                labels.append(name)
                bboxes.append([xmin, ymin, xmax, ymax])

            # Create a dictionary with the data for the file
            file_data = {'filename': file, 'labels': labels, 'bboxes': bboxes}

            # Append the dictionary to the data list
            annotations.append(file_data)

    for image in os.listdir(image_path):
        if image.endswith(".png"):
            images.append(PIL.Image.open(os.path.join(image_path, image)))

    return annotations, images


def split_dataset_into_train_val_test(dataset, split=(0.8, 0.1)):
    # Split the dataset into train, validation, and test
    train = dataset[:int(len(dataset) * split[0])]
    remaining = dataset[int(len(dataset) * split[0]):]
    val = remaining[:int(len(remaining) * split[1])]
    test = remaining[int(len(remaining) * split[1]):]
    return train, val, test


def custom_dump_images_and_labels(annotations, images, set_name):
    for i, example in enumerate(annotations):
        image = images[i]
        labels = example['labels']
        bbox = example['bboxes']
        targets = []
        image_width, image_height = image.size
        for label, box in zip(labels, bbox):
            bbox_width = box[2] - box[0]
            bbox_height = box[3] - box[1]
            x_center = box[0] + bbox_width / 2
            y_center = box[1] + bbox_height / 2

            # Normalise the values
            x_center /= image_width
            y_center /= image_height
            bbox_width /= image_width
            bbox_height /= image_height

            label = label_lookup[label]

            # Add the values to the targets list
            targets.append(f"{label} {x_center} {y_center} {bbox_width} {bbox_height}")

        if set_name == "train":
            root_dir = "train"
        elif set_name == "test":
            root_dir = "test"
        else:
            root_dir = "val"

        with open(dataset_root + "/" + label_root + "/" + root_dir + "/" + str(i) + ".txt", "w") as f:
            for target in targets:
                f.write(target + "\n")
        image.save(dataset_root + "/" + image_root + "/" + root_dir + "/" + str(i) + ".png")



annotations, images = load_data()
trainA, valA, testA = split_dataset_into_train_val_test(annotations)
trainI, valI, testI = split_dataset_into_train_val_test(images)



print("Stats for the dataset:")
print("Train:", len(trainA))
print("Validation:", len(valA))
print("Test:", len(testA))
print("Total:", len(trainA) + len(valA) + len(testA))

# Create dirs
os.makedirs(dataset_root + "/" + image_root + "/" + "train", exist_ok=True)
os.makedirs(dataset_root + "/" + image_root + "/" + "val", exist_ok=True)
os.makedirs(dataset_root + "/" + image_root + "/" + "test", exist_ok=True)
os.makedirs(dataset_root + "/" + label_root + "/" + "train", exist_ok=True)
os.makedirs(dataset_root + "/" + label_root + "/" + "val", exist_ok=True)
os.makedirs(dataset_root + "/" + label_root + "/" + "test", exist_ok=True)



print(trainA[0])
print(trainI[0])
print(valA[0])
print(valI[0])
print(testA[0])
print(testI[0])


custom_dump_images_and_labels(trainA, trainI, "train")
custom_dump_images_and_labels(valA, valI, "val")
custom_dump_images_and_labels(testA, testI, "test")