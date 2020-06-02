


Bounding box formats:

coco: [x_min, y_min, width, height]
pascal_voc: [x_min, y_min, x_max, y_max]


# COCO (Common Object in Context) Data Format

_http://cocodataset.org/#format-data_


```json
{
    "info": info,
    "images": [image],
    "annotations": [annotation],
    "licenses": [license],
}

info{
    "year": int,
    "version": str,
    "description": str,
    "contributor": str,
    "url": str,
    "date_created": datetime,
}

image{
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
    "license": int,
    "flickr_url": str,
    "coco_url": str,
    "date_captured": datetime,
}

license{
    "id": int,
    "name": str,
    "url": str,
}

```


#### Object Detection


```json
annotation{
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon
    ],
    "area": float,
    "bbox": [x,y,width,height
    ],
    "iscrowd": 0 or 1,
}
    
    categories[
    {
        "id": int,
        "name": str,
        "supercategory": str,
    }
]annotation{
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon
    ],
    "area": float,
    "bbox": [x,y,width,height
    ],
    "iscrowd": 0 or 1,
}
    
    categories[
    {
        "id": int,
        "name": str,
        "supercategory": str,
    }
]

```


#### Keypoint Detection

```json
annotation{
    "keypoints": [x1,y1,v1,...
    ],
    "num_keypoints": int,
    "[cloned]": ...,
}
    
categories[
    {
        "keypoints": [str
        ],
        "skeleton": [edge
        ],
        "[cloned]": ...,
    }
]
    
"[cloned]": denotes fields copied from object detection annotations defined above.
```


#### Panoptic Segmentation

```json
annotation{
    "image_id": int,
    "file_name": str,
    "segments_info": [segment_info],
}
        
segment_info{
    "id": int,. "category_id": int,
    "area": int,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}
        
categories[
    {
        "id": int,
        "name": str,
        "supercategory": str,
        "isthing": 0 or 1,
        "color": [R,G,B],
    }
]
```
