import base64 
import sys
import json
import read_image as ri 

path_to_text='../files.txt'
f = open(path_to_text).read()
f = f.split('\n')
f = f[:-1]
f = list(map(lambda x: '/Volumes/SUSB/OAS2_RAW_PART2/'+x+'/RAW/', f))
filenames = ['mpr-1.nifti.hdr', 'mpr-2.nifti.hdr', 'mpr-3.nifti.hdr']
filepath_names = []
test_files = 43

for i in range(test_files):
    filepath_names.append(f[i]+filenames[0])

for filepath in filepath_names:
    img = ri.Image(filepath)
    kdata = img.get_k_data(usfactor=0.5)
    img = base64.b64encode(kdata)
    print(json.dumps({"key":"0", "image_bytes": {"b64": img}}))    
