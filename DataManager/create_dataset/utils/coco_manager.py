import glob, os, json, cv2
import numpy as np
from PIL import Image

def createColorMaskImage(instanceImage, instanceLabels, categoryKey, categoryToColorMapping):

        '''Creates a simplified color image for the instance image using masks
        @param instanceImage: Image with the instanceid
        @param instanceLabels: Result of reading the metadata file
        @param categoryKey: Instance metadata key
        @param categoryToColorMapping: dictionary of categoryvalue --> color
        @returns:Simplified color mapping masks'''

        w = instanceImage.shape[1]
        h = instanceImage.shape[0]
        colorMaskImage = np.zeros((h,w,3), np.uint8)
        uniqueImagePixelValues = np.unique(instanceImage)
        for id in uniqueImagePixelValues:        
            if not id in instanceLabels.keys():
                continue
            if not categoryKey in instanceLabels[id].keys():
                continue
            categoryValue = instanceLabels[id][categoryKey]
            if (not categoryValue in categoryToColorMapping):
                continue
            color = categoryToColorMapping[categoryValue]
            colorMaskImage[instanceImage==id,0] = color[0]
            colorMaskImage[instanceImage==id,1] = color[1]
            colorMaskImage[instanceImage==id,2] = color[2] 
        return colorMaskImage

def readInstanceImage(path):
        '''Reads a instance iamage
        @param path: Path to the file
        @return: instance id as a uint'''
        return cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

def load_json(path):
    # Load json from file
    json_file = open(path)
    js = json.load(json_file)
    json_file.close()
    return js

class MaskManager:
    def __init__(self, dataset_path): #dataset_config_path):
        self.dataset_path = dataset_path
        
        self.masks = dict()
        self.super_categories = dict()
        self.custom_classes = dict()
        self.resave_masks = False
        self.resave_images_flag = False
        self.test_train_val_flag = False
        self.custom_classes_flag = False        # used to add a experimental custom class (e.g., instead of fork/spoon all are utensels)
        self.resave_images_path = None
        
        #TODO: need to have a path/folder validator
        self.dataset_raw_folder = os.path.join(dataset_path, 'raw/')
        self.dataset_config_path = os.path.join(dataset_path, 'dataset_config/')

        self.mask_colors = { 
            "fork":(0,255,0), 
            "spoon":(0,255,0), 
            "knife":(0,255,0), 
            'coffeeCup':(255,0,0), 
            'clearCup':(0,0,255)
        }
        self.colorMapping = { 
            "fork":[0,255,0], 
            "spoon":[0,255,0], 
            "knife":[0,255,0], 
            'coffeeCup':[255,0,0], 
            'clearCup':[0,0,255]
        }

        self.categories = ["fork", "spoon", "knife", "coffeeCup", "clearCup"]

        self.super_categories = {
            "fork":"fork", 
            "spoon":"spoon", 
            "knife":"knife", 
            'coffeeCup':'coffeeCup', 
            'clearCup':'clearCup'}
        self.get_super_categories = {
            "fork":["fork"], 
            "spoon":["spoon"], 
            "knife":["knife"], 
            #'ms_utensils': ['fork', 'spoon', 'knife'],
            'coffeeCup': ['coffeeCup'],
            'clearCup':['clearCup']
        }

        
    def set_resave_mask(self, new_mask_path="/home/redne/mnt/project_zero/project_zero/ds1/masks/"):
        self.resave_masks = True
        self.resave_mask_path = new_mask_path

    def set_test_train_val_flag(self):
        self.test_train_val_flag = True

    def add_category(self, category, super_category):
        """ Adds a new category to the set of the corresponding super_category
        Args:
            category: e.g. 'eagle'
            super_category: e.g. 'bird'
        Returns:
            True if successful, False if the category was already in the dictionary
        """
        if not self.super_categories.get(super_category):
            # Super category doesn't exist yet, create a new set
            self.super_categories[super_category] = {category}
        elif category in self.super_categories[super_category]:
            # Category is already accounted for
            return False
        else:
            # Add the category to the existing super category set
            self.super_categories[super_category].add(category)

        return True # Addition was successful

    def add_mask(self, image_path, mask_path, color_categories):
        """ Takes an image path, its corresponding mask path, and its color categories,
            and adds it to the appropriate dictionaries
        Args:
            image_path: the relative path to the image, e.g. './images/00000001.png'
            mask_path: the relative path to the mask image, e.g. './masks/00000001.png'
            color_categories: the legend of color categories, for this particular mask,
                represented as an rgb-color keyed dictionary of category names and their super categories.
                (the color category associations are not assumed to be consistent across images)
        Returns:
            True if successful, False if the image was already in the dictionary
        """
        if self.masks.get(image_path):
            return False # image/mask is already in the dictionary

        # Create the mask definition
        mask = {
                
            'mask': mask_path,
            'color_categories': color_categories
        }

        # Add the mask definition to the dictionary of masks
        self.masks[image_path] = mask

        # Regardless of color, we need to store each new category under its supercategory
        for _, item in color_categories.items():
            self.add_category(item['category'], item['super_category'])

        return True # Addition was successful

    def parseInstancelabels(self, path):
            
        '''
        sx_parseutils
        Reads the instance json that describes each unique instance and the corresponding metadata. 
        @param path: Path to the file
        @return: Returns a directory of key=pixelvalue value=metadata (a dictionary of# metadata key and value)'''
        instanceMetadataDictionaryByInstanceId = {}
        with open(path) as inJsonText:   
            #reads the event (frame) json        
            inJson = json.load(inJsonText)  
            #sanity check
            if "InstanceUniqueIds" not in inJson['Events'][0]['Event']['Data']: 
                raise RuntimeError('BAD JSON; This is not an instance GT folder') 
            # Makes an instancename --> metadata array
            instanceMetadataDictionaryByName = {}
            for dataArray in inJson['Events'][0]['Event']['Data']["Arr"]:
                instanceMetadata = {}
                for metadata in dataArray["Metadata"]["MapEntries"]:
                    instanceMetadata[metadata['Key']]=metadata['Val']
                instanceMetadataDictionaryByName[instanceMetadata['Name']] = instanceMetadata
            # Re-Store the metadata lookup based on pixelvalue.
            for index, instanceFullName in enumerate(inJson['Events'][0]['Event']['Data']['InstanceUniqueIds']):
                name = instanceFullName.split('/')[-1]
                if name in instanceMetadataDictionaryByName: 
                    instanceMetadataDictionaryByInstanceId[index] = instanceMetadataDictionaryByName[name]        
        return instanceMetadataDictionaryByInstanceId

    def createColorMaskImage(self, instanceImage, instanceLabels, categoryKey, categoryToColorMapping):
        '''
        sx_parseutils
        Creates a simplified color image for the instance image using masks
        @param instanceImage: Image with the instanceid
        @param instanceLabels: Result of reading the metadata file
        @param categoryKey: Instance metadata key
        @param categoryToColorMapping: dictionary of categoryvalue --> color
        @returns:Simplified color mapping masks'''
        w = instanceImage.shape[1]
        h = instanceImage.shape[0]
        colorMaskImage = np.zeros((h,w,3), np.uint8)
        uniqueImagePixelValues = np.unique(instanceImage)
        for id in uniqueImagePixelValues:        
            if not id in instanceLabels.keys():
                continue
            if not categoryKey in instanceLabels[id].keys():
                continue
            categoryValue = instanceLabels[id][categoryKey]
            if (not categoryValue in categoryToColorMapping):
                continue
            color = categoryToColorMapping[categoryValue]
            colorMaskImage[instanceImage==id,0] = color[0]
            colorMaskImage[instanceImage==id,1] = color[1]
            colorMaskImage[instanceImage==id,2] = color[2] 
        return colorMaskImage
    

    def resave_mask_png(self, instanceImagePath, instanceLabels, instance_name):
        """
        (re) save just mask (non-depth) forground  mask objects
        in cases where the _DEGUG_mask.png is used from DS0,1,2 iterations, this is a correction for a 
        fresh mask png to be saved without the rgb gray background
        
        """
        instanceImage = cv2.imread(instanceImagePath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        #rgbImage = cv2.imread(rgbImagePath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        #grayImage = cv2.cvtColor(cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)

        outMaskImageForViewing = self.createColorMaskImage(instanceImage, instanceLabels, 'CategoryPath', self.colorMapping)

        outMask_path = os.path.join(self.resave_mask_path,instance_name+"_mask_.png" )
        cv2.imwrite(outMask_path, outMaskImageForViewing)
        return outMask_path
        
    def old_make_datapath_list(self, path, sample_amount=0.1):
        mask_paths = glob.glob(path + "*mask.png")
        masklen = len(mask_paths)-1
        #masklen=10
        mask_paths = mask_paths[:masklen+1]

        sample =  round(masklen * sample_amount)

        train_, val_ = mask_paths[-masklen+sample:], mask_paths[:-masklen+sample]
        
        return train_, val_

    def make_datapath_list(self, path, sample_amount=0.1):
        mask_paths = glob.glob(path + "*mask.png")
        masklen = len(mask_paths)-1
        #masklen=10
        mask_paths = mask_paths[:masklen+1]

        sample =  round(masklen * sample_amount)

        train_, val_ = mask_paths[-masklen+sample:], mask_paths[:-masklen+sample]
        
        return train_, val_

    def start(self,phase, mask_paths):
        root = '/home/redne/mnt/project_zero/project_zero/ds1/parsed/'
        raw = '/home/redne/mnt/project_zero/project_zero/ds1/raw/'
        mask_instance_event_file = '/instance/events.0.json'

        for instance_ in mask_paths:
            #instance_name = os.path.basename(instance_)[:-15]
            instance_name = instance_
            
            #rgb_path = root + instance_name + "_rgb.jpg"
            rgb_path= os.path.join(self.dataset_raw_folder, instance_name, "rgba/outputs")
            rgb_path = glob.glob(rgb_path + "/*.png")[0]
            if self.resave_images_flag == True:
                rgbImage = cv2.imread(rgb_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                rgb_path =os.path.join(self.resave_images_path, instance_name + "_rbg.jpg")
                cv2.imwrite(rgb_path, rgbImage)

            

            #mask_path = root + instance_name + "_DEBUG_mask.png"

            #iter_, instance_base = instance_name.split('_', 1)
            #instance_name_raw = instance_base + '_' +iter_

            #mask_event = raw + instance_name_raw + mask_instance_event_file
            mask_event = os.path.join(self.dataset_raw_folder, instance_name, 'instance/events.0.json')
            
            #instanceImagePath = raw + instance_name_raw + '/instance/outputs'
            #instanceImagePath = glob.glob(instanceImagePath + "/*.exr")[0]

            instanceImagePath  = os.path.join(self.dataset_raw_folder, instance_name, "instance/outputs")
            instanceImagePath = glob.glob(instanceImagePath + "/*.exr")[0]
            
            instanceLabels = self.parseInstancelabels(mask_event)

            color_categories = dict()

            for id in instanceLabels.keys():        
                for label in self.categories:
                    if label in instanceLabels[id]['Name']:
                        if self.custom_classes_flag == True: 
                            label = self.custom_classes[label]
                        instanceLabels[id]['CategoryPath'] = label
                        self.add_category(label,self.super_categories[label])
                        color_categories[str(self.mask_colors[label])] = {
                            "category": label,
                            "super_category": self.super_categories[label]
                        }
            
            if self.resave_masks == True: 
                instanceImage = readInstanceImage(instanceImagePath)
                
                outMaskImageForViewing = createColorMaskImage(instanceImage, instanceLabels, 'CategoryPath', self.colorMapping)
                mask_path = os.path.join(self.resave_mask_path,instance_name+"_mask.png" )
                cv2.imwrite(mask_path, outMaskImageForViewing) ## uncomment if new dataset

            self.add_mask(
                rgb_path,
                mask_path, 
                color_categories
            )
            

    #def write_mask_to_json(self):
    def get_masks(self):
        return self.masks

    def write_masks_to_json(self,phase):
        """ Writes all masks and color categories to the output file path as JSON
        """
        # Serialize the masks and super categories dictionaries
        serializable_masks = self.get_masks()
        serializable_super_cats = self.get_super_categories
        masks_obj = {
            'masks': serializable_masks,
            'super_categories': serializable_super_cats
        }

        # Write the JSON output file
        output_file_path = self.dataset_config_path + f'{phase}_mask_definitions.json'
        with open(output_file_path, 'w+') as json_file:
            json_file.write(json.dumps(masks_obj))

    def show_mask_img(self, index_):
        image_mask_path = self.masks[list(self.masks.keys())[index_]]['mask']
        print(f'quering image masks from: {image_mask_path}')
        mask_image = Image.open(image_mask_path)
        mask_image = mask_image.convert("RGB")
        return mask_image
