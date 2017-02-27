import os

def update_labels_file():
    os.remove("retrained_labels.txt")

    labels = os.listdir("../dataset")

    with open("retrained_labels.txt", "w") as labels_file:
        for line in labels:
            labels_file.write(line + "\n")