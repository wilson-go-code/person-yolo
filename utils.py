import json

def load_json(filepath):
    with open(filepath, "r") as buffer:
        data = json.load(buffer)

    return data

def filter_json(label):
    ground_truth = {}
    for index, item in enumerate(label["images"]):
        ground_truth[item["id"]] = {
            "filename": item["file_name"], 
            "height": item["height"], 
            "width": item["width"]
        }

    for index, item in enumerate(label["annotations"]):
        if item["image_id"] in ground_truth and item["category_id"] == 1:
            if "bbox" not in ground_truth[item["image_id"]]:
                ground_truth[item["image_id"]]["bbox"] = [item["bbox"]]
            else:
                ground_truth[item["image_id"]]["bbox"].append(item["bbox"])

    return ground_truth

def convert_pixel_to_yolo(bbox, size):
    x, y, w, h = bbox
    height, width = size[0], size[1]
    xhat = (x + w / 2) / width
    yhat = (y + h / 2) / height
    what = w / width
    hhat = h / height

    return [0, xhat, yhat, what, hhat]

def convert_yolo_to_pixel(fmtyolo, size):
    height, width = size
    _, xhat, yhat, what, hhat= fmtyolo
    w = int(what * width)
    h = int(hhat * height)
    x = int(xhat * width - w / 2)
    y = int(yhat * height - h / 2)

    return [x, y, w, h]

def save_yolo(ground_truth, savedir):
    for key, image in ground_truth.items():
        filename = image["filename"].split('.')[0]
        bbox = image["bbox"]
        size = [image["height"], image["width"]]
        with open(savedir + filename + ".txt", "a") as file:
            for item in bbox:
                file.write(' '.join(str(x) for x in convert_pixel_to_yolo(item, size)))
                file.write('\n')

def process_label(labelpath, savedir):
    label = load_json(labelpath)
    ground_truth = filter_json(label)
    save_yolo(ground_truth, savedir)
