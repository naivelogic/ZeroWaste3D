import json
import glob

result = []
for f in glob.glob('/home/redne/ZeroWaste3D/DataManager/create_dataset/sample_maya_raw/ds1/parsed1/' +"*.json"):
    with open(f, "r") as infile:
        result += json.load(infile)

with open("./merge.json", "w") as outfile:
    json.dump(result, outfile)