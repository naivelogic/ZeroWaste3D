
CATEGORY_LIST = ["clearCup", "coffeeCup", "fork", "spoon", "knife"]

#cat_id
CATEGORY_IDS = {
    'clearCup': 1,
    'coffeeCup': 2,
    'fork': 3,
    'spoon':4,
    'knife':5,
}

#categories
COCO_CATEGORIES = [
      {
         "supercategory":"clearCup",
         "id":1,
         "name":"clearCup"
      },
    {
         "supercategory":"coffeeCup",
         "id":2,
         "name":"coffeeCup"
      },
    {
         "supercategory":"ms_utensils",
         "id":3,
         "name":"fork"
      },
      {
         "supercategory":"ms_utensils",
         "id":4,
         "name":"spoon"
      },
      {
         "supercategory":"ms_utensils",
         "id":5,
         "name":"knife"
      },
    
   ]

#info
COCO_INFO = {
    "description": "MSFT Synthetics ZeroWaste Project - Dataset 2",
    "url": "",
    "version": "2",
    "year": 2020,
    "contributor": "MSFT Synthetics",
    "date_created": "10/19/2020"
}

#licenses
COCO_LICENSES = [{
    "url": "",
    "id": 0,
    "name": "License"
}]


#zerowaste_ds2_process_error_files = ['iter120','iter121','iter122','iter123','iter124','iter125','iter126','iter127','iter128','iter129','iter150','iter151','iter152','iter153','iter154','iter155','iter156','iter157','iter158','iter159','iter190','iter191','iter192','iter193','iter194','iter195','iter196','iter197','iter198','iter199','iter240','iter241','iter242','iter243','iter244','iter245','iter26','iter247','iter248','iter249','iter380','iter381','iter382','iter383','iter384','iter385','iter386','iter387','iter388','iter389','iter70','iter71','iter72','iter73','iter74','iter75','iter76','iter77','iter78','iter79']
#for dataDirectory in dataDirectories:
#        if dataDirectory[0] in zerowaste_ds2_process_error_files:
#            continue
#        process_folder([dataDirectory, outPathRoot])