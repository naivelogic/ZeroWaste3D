"""
utilitis and functions to assist with custom preprocessing task for objects in DS1+ (created Nov 2020)
"""

ds1_storm_custom_filter = {
    'S_cup': 3000,
    'P_foodcontainer': 4000,
    'P_beveragecontainer': 9000,
    'M_aerosol': 2300,
    'H_otherbottle': 8000,
    'H_facemask': 3000,
    'H_beveragebottle': 3000,
    'D_lid': 5000,
    'D_foodcontainer': 9000,
    'P_cup': 3000,
}

CATEGORY_LIST = ['BG', "H_beveragebottle", "D_lid", "S_cup", "P_foodcontainer", "P_beveragecontainer", "D_foodcontainer",
                 "H_facemask", "M_aerosol", "H_otherbottle", 'P_cup']

def custom_annotation_filter(annotations):
    annie = []
    counter_ = 0
    for ta in annotations:
        thresh = ds1_storm_custom_filter[CATEGORY_LIST[ta['category_id']]]
        if ta['area'] > thresh:
            annie.append(ta)
        else:
            #print("exception")
            counter_ +=1
    print(f"{counter_} # of objects filtered bc bbox and pixels too small" )

    return annie
    
