The cloud-ml files. These are based off of the [flowers](https://cloud.google.com/ml-engine/docs/flowers-tutorial) tutorial and corresponding code.

Goal: Perform [compressed sensing reconstruction](https://en.wikipedia.org/wiki/Compressed_sensing) using machine learning.

Methods: I used the [inception network](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) along with the [google cloud platform](https://cloud.google.com/) to train the network. The google cloud platform gives a free trial, which is why I used it.

Quickstart: Define the following environment variables:

JOB_NAME -- the name of the job to submit to the cloud

OUTPUT_PATH -- the path to output data, should be on the google cloud servers

REGION -- the region the google cloud servers are located to run the model

TRAIN_DATA -- the training data path, should be on the google cloud servers

EVAL_DATA -- the evaluation data path, should be on the google cloud servers

Use the following command:

`gcloud ml-engine jobs submit training $JOB_NAME     --job-dir $OUTPUT_PATH     --runtime-version 1.4     --module-name trainer.task     --package-path trainer/     --region $REGION    --config config.yaml      --     --train_data_paths $TRAIN_DATA     --eval_data_paths $EVAL_DATA   --output_path $OUTPUT_PATH      --verbosity DEBUG`
