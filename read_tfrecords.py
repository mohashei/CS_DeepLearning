import tensorflow as tf
import numpy as np

reconstructed_images = []

record_iterator = tf.python_io.tf_record_iterator(path='mri_mprage.tfrecords')

for string_record in record_iterator:
    
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    height = int(example.features.feature['height']
                                 .int64_list
                                 .value[0])
    
    width = int(example.features.feature['width']
                                .int64_list
                                .value[0])
    
    img_string = (example.features.feature['kspace']
                                  .bytes_list
                                  .value[0])
    
    annotation_string = (example.features.feature['xspace']
                                .bytes_list
                                .value[0])

    img_1d = np.fromstring(img_string, dtype=np.float64)
    print(img_1d.shape)
