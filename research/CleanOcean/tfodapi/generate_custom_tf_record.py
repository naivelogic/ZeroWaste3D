"""
Sample from raccoon dataset by datitran - https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py

Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""

import io
import os
from collections import namedtuple

import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from tqdm import tqdm

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'D_lid':
        return 1
    elif row_label == 'H_beveragebottle':
        return 2
    elif row_label == 'S_cup':
        return 3
    else:
    # classes removed: 'lunch box'; 'cell phone'; 'aluminum foil'
        None


def split(df, group):
    """Group all bounding box annotations on one image together by the filename of the image.

    Args:
        df: training or test split labels CSV read as a pandas DataFrame
        group: which column to use as the groupby key

    Returns:

    """
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    grouped = [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
    return grouped


def create_tf_example(group, image_dir):
    image_path = os.path.join(image_dir, '{}'.format(group.filename))
    if not os.path.exists(image_path):
        print('Image {} is not found; skipped.'.format(image_path))
        return None

    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        class_int = class_text_to_int(row['class'])
        if class_int is None:  # skip bounding box if category is not in the accepted list of categories
            continue

        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_int)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    print('Creating tfrecord examples for {} images, with {} bounding box annotations.'.format(
        len(grouped), len(examples)))

    num_examples = 0
    for group in tqdm(grouped):
        tf_example = create_tf_example(group, FLAGS.image_dir)

        if tf_example is not None:  # if the image exists
            writer.write(tf_example.SerializeToString())
            num_examples += 1

    writer.close()
    print('Successfully created the TFRecords: {}. Number of examples included is {}.'.format(
        FLAGS.output_path, num_examples))


if __name__ == '__main__':
    tf.app.run()