import read_image as ri 
import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tfrecords_filename = 'mri_mprage.tfrecords'

options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
writer = tf.python_io.TFRecordWriter(tfrecords_filename, options=options)

path_to_text='../files.txt'
f = open(path_to_text).read()
f = f.split('\n')
f = f[:-1]
f = list(map(lambda x: '/Volumes/SUSB/OAS2_RAW_PART2/'+x+'/RAW/', f))
filenames = ['mpr-1.nifti.hdr', 'mpr-2.nifti.hdr', 'mpr-3.nifti.hdr']
filepath_names = []
eval_files = 10
tot_files = 32
for i in range(tot_files):
    filepath_names.append(f[i]+filenames[0])
    filepath_names.append(f[i]+filenames[1])
    filepath_names.append(f[i]+filenames[2])

for filepath in filepath_names:
    img = ri.Image(filepath)
    kdata = img.get_k_data(usfactor=0.1)
    xdata = img.get_x_data()
    for z in range(kdata.shape[1]):
        kdata_slice = np.squeeze(kdata[:,z])
        xdata_slice = np.squeeze(xdata[:,:,z])
        width = xdata_slice.shape[0]
        height = xdata_slice.shape[1]
        kdata_slice = np.concatenate((np.real(kdata_slice),
                                      np.imag(kdata_slice)), axis=0) 
        xdata_slice = xdata_slice.reshape(width * height)
        xdata_slice = np.squeeze(xdata_slice)
        kdata_slice_raw = kdata_slice.tostring()
        xdata_slice_raw = xdata_slice.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'kspace': _bytes_feature(kdata_slice_raw),
        'xspace': _bytes_feature(xdata_slice_raw)}))
        writer.write(example.SerializeToString())

writer.close()
