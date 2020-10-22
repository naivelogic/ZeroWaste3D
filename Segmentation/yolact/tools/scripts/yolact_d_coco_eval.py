"""
Modified based on: https://github.com/dbolya/yolact
Runs the coco-supplied cocoeval script to evaluate detections
outputted by using the output_coco_json flag in eval.py.

python run_coco_eval.py --gt_ann_file /mnt/omreast_users/phhale/zerowaste/02-datasets/ds2/coco_ds_3class/test_coco_instances.json
"""


import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

#pip install terminaltables

parser = argparse.ArgumentParser(description='COCO Detections Evaluator')
parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str)
parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str)
#parser.add_argument('--gt_ann_file',   default='data/coco/annotations/instances_val2017.json', type=str)
parser.add_argument('--gt_ann_file',   default='/mnt/omreast_users/phhale/zerowaste/02-datasets/ds2/coco_ds_3class/test_coco_instances.json', type=str)
parser.add_argument('--eval_type',     default='both', choices=['bbox', 'mask', 'both'], type=str)
args = parser.parse_args()



if __name__ == '__main__':
	"""
	from argparse import Namespace

	args = Namespace(
		bbox_det_file = 'results/bbox_detections.json',
		mask_det_file = 'results/mask_detections.json',
		gt_ann_file = '/mnt/omreast_users/phhale/zerowaste/02-datasets/ds2/coco_ds_3class/test_coco_instances.json',
		eval_type = 'both'
	)

	print(args)
	"""
	eval_bbox = (args.eval_type in ('bbox', 'both'))
	eval_mask = (args.eval_type in ('mask', 'both'))

	print('Loading annotations...')
	gt_annotations = COCO(args.gt_ann_file)
	coco = gt_annotations
	if eval_bbox:
		bbox_dets = gt_annotations.loadRes(args.bbox_det_file)
	if eval_mask:
		mask_dets = gt_annotations.loadRes(args.mask_det_file)

	if eval_bbox:
		print('\nEvaluating BBoxes:')
		bbox_eval = COCOeval(gt_annotations, bbox_dets, 'bbox')
		bbox_eval.evaluate()
		bbox_eval.accumulate()
		bbox_eval.summarize()
	
	print("\nPer-category bbox AP:")
	precisions = bbox_eval.eval['precision']
	catIds = coco.getCatIds() # same as bbox_eval.eval['params'].catIds and gt_annotations.cats
	# precision has dims (iou, recall, cls, area range, max dets)
	assert len(catIds) == precisions.shape[2]
	results_per_category = []

	for idx, catId in enumerate(catIds):
		# area range index 0: all area ranges
		# max dets index -1: typically 100 per image
		nm = coco.loadCats(catId)[0]
		precision = precisions[:, :, idx, 0, -1]
		precision = precision[precision > -1]
		ap = np.mean(precision) if precision.size else float('nan')
		results_per_category.append(
			('{}'.format(nm['name']),
				'{:0.3f}'.format(float(ap * 100))))

	#https://github.com/josephkokchin/_isabella/blob/master/mmdet/core/evaluation/coco_utils.py
	import itertools
	from terminaltables import AsciiTable
	N_COLS = min(6, len(results_per_category) * 2)
	results_flatten = list(itertools.chain(*results_per_category))
	headers = ['category', 'AP'] * (N_COLS // 2)
	results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
	table_data = [headers]
	table_data += [result for result in results_2d]
	table = AsciiTable(table_data)
	print(table.table)

	if eval_mask:
		print('\nEvaluating Masks:')
		bbox_eval = COCOeval(gt_annotations, mask_dets, 'segm')
		bbox_eval.evaluate()
		bbox_eval.accumulate()
		bbox_eval.summarize()

	print("\nPer-category segm AP: ")
	precisions = bbox_eval.eval['precision']
	catIds = coco.getCatIds() # same as bbox_eval.eval['params'].catIds and gt_annotations.cats
	# precision has dims (iou, recall, cls, area range, max dets)
	assert len(catIds) == precisions.shape[2]
	results_per_category = []

	for idx, catId in enumerate(catIds):
		# area range index 0: all area ranges
		# max dets index -1: typically 100 per image
		nm = coco.loadCats(catId)[0]
		precision = precisions[:, :, idx, 0, -1]
		precision = precision[precision > -1]
		ap = np.mean(precision) if precision.size else float('nan')
		results_per_category.append(
			('{}'.format(nm['name']),
				'{:0.3f}'.format(float(ap * 100))))

	N_COLS = min(6, len(results_per_category) * 2)
	results_flatten = list(itertools.chain(*results_per_category))
	headers = ['category', 'AP'] * (N_COLS // 2)
	results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
	table_data = [headers]
	table_data += [result for result in results_2d]
	table = AsciiTable(table_data)
	print(table.table)