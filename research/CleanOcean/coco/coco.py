from collections import defaultdict
import cv2
from functools import partial
import json
from multiprocessing import Pool
import os
import re

categories = ['ancient_artillery', 'bandit_archer', 'bandit_guard', 'black_imp', 'cave_bear', 'city_archer', 'city_guard', 'cultist', 'deep_terror', 'earth_demon', 'flame_demon', 'forest_imp', 'frost_demon', 'giant_viper', 'harrower_infester', 'hound', 'inox_archer', 'inox_guard', 'inox_shaman', 'living_bones', 'living_corpse', 'living_spirit', 'lurker', 'night_demon', 'ooze', 'savvas_icestorm', 'savvas_lavaflow', 'spitting_drake', 'stone_golem', 'sun_demon', 'vermling_scout', 'vermling_shaman', 'vicious_drake', 'wind_demon']
is_crowd = 0

MASK_REGEX = r"image_\d+_mask_\d+_(.*)\.png"


def get_mask_contours(image_id, image_masks, debug=False):
    image_id_filter = image_id + "_"
    image_masks = [x for x in image_masks if image_id_filter in x[1]]

    result = []

    if debug:
        print("{}: {}".format(image_id, image_masks))

    mask_id = 0
    for (dirpath, filename) in image_masks:
        mask = cv2.imread(os.path.join(dirpath, filename))
        name = re.match(MASK_REGEX, filename)[1]

        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = sorted(contours, key=lambda x: -cv2.contourArea(x))[0]

        result.append((name, contour))

        if debug:
            cv2.drawContours(mask, contours, -1, (0, 255, 0), thickness=1)
            cv2.drawContours(mask, [contour], -1, (0, 0, 255), thickness=2)
            cv2.imshow("{}: {} - {}".format(image_id, mask_id, name), mask)

        mask_id += 1

    if debug:
        print([x[0] for x in result])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    segmentations = [cv2.approxPolyDP(sub_mask, 1, True)]
    bbox = cv2.boundingRect(segmentations[0])

    return {
        'segmentation': [s.ravel().tolist() for s in segmentations],
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': tuple(bbox),
        'area': cv2.contourArea(sub_mask)
    }


def process_image(image, image_masks):
    dirpath, filename = image
    image_id = filename.replace(".png", "")

    try:
        sub_masks = get_mask_contours(image_id, image_masks)
    except IndexError:
        print("Error processing image: " + image_id)
        return None

    annotations = []
    annotation_id = 1
    for (name, sub_mask) in sub_masks:
        category_id = categories.index(name) + 1
        annotation = create_sub_mask_annotation(
            sub_mask,
            image_id,
            category_id,
            image_id + "_" + str(annotation_id),
            is_crowd
        )
        annotations.append(annotation)
        annotation_id += 1

    return {
        'annotations': annotations,
        'image': {
            'id': image_id,
            # 'file_name': os.path.join(dirpath, image_id).replace("masks", "images") + ".png"
            'file_name': image_id + ".png"
        }
    }


if __name__ == "__main__":
    mask_directory = r"E:\Generated\masks"
    masks = []

    for (dirpath, dirnames, filenames) in os.walk(mask_directory):
        for filename in filenames:
            if 'mask' not in filename:
                continue

            masks.append((dirpath, filename))

    image_directory = r"E:\Generated\images"
    image_names = []

    for (dirpath, dirnames, filenames) in os.walk(image_directory):
        if "val" in dirpath:
            print("Skipping val directory")
            continue

        for filename in filenames:
            image_names.append((dirpath, filename))

    processed_images = []
    """
    for idx in range(10):
        processed_images.append(process_image(image_names[idx], masks))
    """

    with Pool(5) as pool:
        processed_images.extend(pool.map(partial(process_image, image_masks=masks), image_names))

    annotations = []
    images = []
    all_monsters_found = defaultdict(lambda: 0)
    for el in processed_images:
        if el is None:
            continue

        images.append(el["image"])
        annotations.extend(el["annotations"])

        for annotation in el["annotations"]:
            all_monsters_found[categories[annotation["category_id"] - 1]] += 1

    print("")
    print("")
    print("{} successful images".format(len(images)))

    print("\n\n== Monsters found: {} ==".format(len(list(all_monsters_found.keys()))))
    for key, value in all_monsters_found.items():
        print("  - {}: {}".format(key, value))

    coco = {
        'categories': [
            {
                'id': idx + 1,
                'name': cat,
                'supercategory': None
            }
            for idx, cat in enumerate(categories)
        ],
        'images': images,
        'annotations': annotations
    }

    with open(r"E:\Generated\annotations.json", "w") as file:
        json.dump(coco, file)