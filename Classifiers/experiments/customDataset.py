from fastai.vision import ImageList
from fastai.vision import imagenet_stats

CROPS_V2 = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/crop_ds0/v2/all_crops/'
def get_csiro_dataset(dataset_path:str=CROPS_V2, train_mode:bool=True, tfms=None, bs:int=32, sz=224, 
                 padding_mode:str='reflection', seed:int=None, split_pct:float=0.2, print_ds_stats:bool=False):
    if train_mode:     
        data = (ImageList.from_folder(dataset_path)
                    .split_by_rand_pct(0.2)
                    .label_from_folder()
                    .transform(tfms, size=sz, padding_mode=padding_mode)
                    .databunch(bs=bs).normalize(imagenet_stats)
                    )
        

    else:    
        from fastai.vision import crop_pad
        data = (ImageList.from_folder(dataset_path)
                        .split_none()
                        .label_from_folder()
                        .transform([crop_pad(), crop_pad()], size=sz, padding_mode=padding_mode)
                        .databunch(bs=bs).normalize(imagenet_stats)
                        )

    if print_ds_stats:
        show_dataset_stats(data)
        
    return data

def show_dataset_stats(data):
    print("------ Data Specifications ------")
    print(data)

    print("------ Data Set Specifications ------")
    print("Number of train images:  ", len(data.train_ds.x))
    print("Number of test images :  ",len(data.valid_ds.x))
    print("Number of image folders: ",len(data.classes))
    print(data.classes)


def get_confusion_matrix(img_input_size, learn):

    MNT_TEST_SET ='/mnt/omreast_users/phhale/csiro_trashnet/original_samples/ValidationVideo/crops_ds2_only'
    test_data = get_csiro_dataset(dataset_path=MNT_TEST_SET, tfms=None, bs=8, sz=img_input_size,
                                    train_mode=False)

    from fastai.train import ClassificationInterpretation
    learn.data.test_dl = test_data.train_dl
    interp_test = ClassificationInterpretation.from_learner(learn=learn, ds_type=3)
    interp_test.plot_confusion_matrix(figsize = (25,5))
    return interp_test


    
        

        