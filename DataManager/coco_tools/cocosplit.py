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

        #number_of_images = len(images)
        # DS1 needed to custom filter synth obj until we updated custom pipeline for edge cases and outliers 
        # ne to normalize a better distribution of bbox and obj locatino during synth rendering pipeline
        #annotations = custom_annotation_filter(annotations)        # custom coco_ds_v2
    
        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        save_coco(args.train, info, licenses, images, filter_annotations(annotations, images), categories)  # custom coco_ds_v2 comment remainder below (uncomment custom_annotaion_filter )

        #x, y = train_test_split(images, train_size=args.split, random_state=1234)

        #save_coco(args.train, info, licenses, x, filter_annotations(annotations, x), categories)
        #save_coco(args.test, info, licenses, y, filter_annotations(annotations, y), categories)
             
        #print("Saved {} entries in {} and {} in {}".format(len(x), args.train, len(y), args.test))


if __name__ == "__main__":
    
    #CSIRO ds1_storm_v3 (updated with P_cup - 11/11/20) 82 # of objects filtered bc bbox and pixels too small - 11/08/20 Train = 791 / test = 178 / val = 20 synthetics (going to val on real)
    #python cocosplit.py --having-annotations -s 0.8 /mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1_storm/coco_ds_v3/coco_instances_v3_original.json /mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1_storm/coco_ds_v3/coco_instances.json /mnt/omreast_users/phhale/csiro_trashnet/datasets/ds1_storm/coco_ds_v3/NULL.json

    #python cocosplit.py --having-annotations -s 0.8 /mnt/omreast_users/phhale/csiro_trashnet/original_samples/Validation_v0/ThreeCategories_TwoCountries_Trashnet/TrashNet.json /mnt/omreast_users/phhale/csiro_trashnet/datasets/crop_ds0/v2/TrashNet.json /mnt/omreast_users/phhale/csiro_trashnet/datasets/crop_ds0/v2/NUL.json

    main(args)