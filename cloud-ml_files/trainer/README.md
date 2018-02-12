The tensorflow files. There are three files, each serving a different task.

`task.py -- the initial file to submit tasks to the cloud. The file is more or less a wrapper for the model.py file, and serves as a way to interact with the model.py file.

model.py -- the main file, which contains the ML model. Some arguments from task.py have defaults defined in model.py.

util.py -- the mostly generic file based on Google's implementation. However, because I am using L2 distance, I had to modify this file.`
