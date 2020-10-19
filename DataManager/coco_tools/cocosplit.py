#https://github.com/akarazniewicz/cocosplit/blob/master/cocosplit.py
# $ python cocosplit.py --having-annotations -s 0.8 /path/to/your/coco_annotations.json train.json test.json

import json
import argparse
import funcy
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('train', type=str, help='Where to store COCO training annotations')
parser.add_argument('test', type=str, help='Where to store COCO test annotations')
parser.add_argument('-s', dest='split', type=float, required=True,
                    help="A percentage of a split; a number in (0, 1)")
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignore all images without annotations. Keep only these with at least one annotation')

args = parser.parse_args()

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def main(args):
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        x, y = train_test_split(images, train_size=args.split)

        save_coco(args.train, info, licenses, x, filter_annotations(annotations, x), categories)
        save_coco(args.test, info, licenses, y, filter_annotations(annotations, y), categories)

        print("Saved {} entries in {} and {} in {}".format(len(x), args.train, len(y), args.test))


if __name__ == "__main__":
    #$ python cocosplit.py --having-annotations -s 0.8 /path/to/your/coco_annotations.json train.json test.json
    #python cocosplit.py --having-annotations -s 0.8 /mnt/omreast_users/phhale/csiro_trashnet/datasets/ds0/coco_ds/coco_instances.json train_coco_instances.json test_coco_instances.json
    #python cocosplit.py --having-annotations -s 0.8 /mnt/omreast_users/phhale/csiro_trashnet/datasets/ds0/coco_ds/coco_instances.json /mnt/omreast_users/phhale/csiro_trashnet/datasets/ds0/coco_ds/train_coco_instances.json /mnt/omreast_users/phhale/csiro_trashnet/datasets/ds0/coco_ds/test_coco_instances.json
    #python cocosplit.py --having-annotations -s 0.9 /mnt/omreast_users/phhale/csiro_trashnet/datasets/ds0/coco_ds/test_coco_instances.json /mnt/omreast_users/phhale/csiro_trashnet/datasets/ds0/coco_ds/test_coco_instances.json /mnt/omreast_users/phhale/csiro_trashnet/datasets/ds0/coco_ds/val_coco_instances.json
    # zerowaste ds1 - 10/19
    #python cocosplit.py --having-annotations -s 0.8 /mnt/omreast_users/phhale/zerowaste/02-datasets/ds2/coco_instances.json /mnt/omreast_users/phhale/zerowaste/02-datasets/ds2/train_coco_instances.json /mnt/omreast_users/phhale/zerowaste/02-datasets/ds2/test_coco_instances.json
    #python cocosplit.py --having-annotations -s 0.9 /mnt/omreast_users/phhale/zerowaste/02-datasets/ds2/test_coco_instances.json /mnt/omreast_users/phhale/zerowaste/02-datasets/ds2/test_coco_instances.json /mnt/omreast_users/phhale/zerowaste/02-datasets/ds2/val_coco_instances.json
    # zerowaste ds1 - 10/19
    #python cocosplit.py --having-annotations -s 0.8 /mnt/omreast_users/phhale/zerowaste/02-datasets/ds2/coco_ds_3class/coco_instances.json /mnt/omreast_users/phhale/zerowaste/02-datasets/ds2/coco_ds_3class/train_coco_instances.json /mnt/omreast_users/phhale/zerowaste/02-datasets/ds2/coco_ds_3class/test_coco_instances.json
    #python cocosplit.py --having-annotations -s 0.9 /mnt/omreast_users/phhale/zerowaste/02-datasets/ds2/coco_ds_3class/test_coco_instances.json /mnt/omreast_users/phhale/zerowaste/02-datasets/ds2/coco_ds_3class/test_coco_instances.json /mnt/omreast_users/phhale/zerowaste/02-datasets/ds2/coco_ds_3class/val_coco_instances.json
    main(args)