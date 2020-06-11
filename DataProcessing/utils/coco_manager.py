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
    def __init__(self, dataset_config_path):
        self.masks = dict()
        self.super_categories = dict()
        self.resave_masks = False
        self.test_train_val_flag = False
        self.dataset_config_path = dataset_config_path

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
        
        self.super_categories_v2 = {
            "fork":"ms_utensils", 
            "spoon":"ms_utensils", 
            "knife":"ms_utensils", 
            'coffeeCup':'coffeeCup', 
            'clearCup':'clearCup'}
        self.get_super_categories_v2 = {
            'ms_utensils': ['fork', 'spoon', 'knife'],
            'coffeeCup': ['coffeeCup'],
            'clearCup':['clearCup']
        }

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

    def add_mask(self, image_path, mask_path, color_categories):
        # Create the mask definition
        mask = {
                
            'mask': mask_path,
            'color_categories': color_categories
        }

        # Add the mask definition to the dictionary of masks
        self.masks[image_path] = mask

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

        outMask_path = os.path.join(self.resave_mask_path,instance_name+"mask_.png" )
        cv2.imwrite(outMask_path, outMaskImageForViewing)
        return outMask_path
        

    def make_datapath_list(self, path):
        mask_paths = glob.glob(path + "*mask.png")
        masklen = len(mask_paths)-1
        #masklen=10
        mask_paths = mask_paths[:masklen+1]

        sample =  round(masklen * .1)

        train_, val_ = mask_paths[-masklen+sample:], mask_paths[:-masklen+sample]
        
        return train_, val_

    def start(self,phase, mask_paths):
        root = '/home/redne/mnt/project_zero/project_zero/ds1/parsed/'
        raw = '/home/redne/mnt/project_zero/project_zero/ds1/raw/'
        mask_instance_event_file = '/instance/events.0.json'

        for instance_ in mask_paths:
            instance_name = os.path.basename(instance_)[:-15]
            rgb_path = root + instance_name + "_rgb.jpg"
            mask_path = root + instance_name + "_DEBUG_mask.png"

            iter_, instance_base = instance_name.split('_', 1)
            instance_name_raw = instance_base + '_' +iter_

            mask_event = raw + instance_name_raw + mask_instance_event_file
            
            instanceImagePath = raw + instance_name_raw + '/instance/outputs'
            instanceImagePath = glob.glob(instanceImagePath + "/*.exr")[0]
            
            instanceLabels = self.parseInstancelabels(mask_event)

            color_categories = dict()

            for id in instanceLabels.keys():        
                for label in self.categories:
                    if label in instanceLabels[id]['Name']:
                        instanceLabels[id]['CategoryPath'] = label 
                        color_categories[str(self.mask_colors[label])] = {
                            "category": label,
                            "super_category": self.super_categories[label]
                        }
            
            if self.resave_masks == True: 
                instanceImage = readInstanceImage(instanceImagePath)
                
                outMaskImageForViewing = createColorMaskImage(instanceImage, instanceLabels, 'CategoryPath', self.colorMapping)
                mask_path = os.path.join(self.resave_mask_path,instance_name+"mask_.png" )
                cv2.imwrite(mask_path, outMaskImageForViewing)

                """
                self.last_mask = {
                    'mask_path':mask_path,
                    'instanceImagePath': instanceImagePath,
                    'instanceImage':instanceImage,
                    'instanceLabels': instanceLabels
                }
                """

                #mask_path = self.resave_mask_png(instanceImagePath, instanceLabels, instance_name)

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
        mask_image = Image.open(image_mask_path)
        mask_image = mask_image.convert("RGB")
        return mask_image
