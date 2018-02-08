Here, we have some IO files for the MRI images. Since we have files of type [nifti](https://nifti.nimh.nih.gov/nifti-1/), we need to use [nibabel](http://nipy.org/nibabel/), which is python's solution for this type of file. Please have this package installed _before_ running anything here! Other than that, please have [numpy](http://www.numpy.org/) and [tensorflow](https://www.tensorflow.org/) installed.

The files are:

read_image.py -- reads the nifti image, generates kspace data from images

convert_to_tfrecords.py -- converts image files into kspace data, and then converts this data into [tfrecords](https://www.tensorflow.org/programmers_guide/datasets) data for use with tensorflow.

show_slices.py -- shows data slice by slice. Nice for visualization of 3D data.

get_kspace.py -- read a single data file, turn it into a json request. Useful for online learning.

read_tfrecords.py -- a test file, which can read tfrecords files.
