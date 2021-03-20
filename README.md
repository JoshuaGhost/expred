# ExPred

This is the implementation of the paper [Explain and Predict, and then Predict Again](https://dl.acm.org/doi/abs/10.1145/3437963.3441758) (accepted in WSDM2021). This code is implemented based on the pipeline model of the [Eraserbenchmark](http://www.eraserbenchmark.com/). All data used by the model can be found from the Eraser Benchmark, too.

## Usage:
  1. Install the required packages from the ```requirements.txt``` by ```pip install -r requirements.txt```
  2. The implementation entry is under ```expred/train```. To run the training, simply copy and paste the following commands:
        ``` export PYTHONPATH=$PYTHONPATH:./ && python expred/train.py --data_dir /dir/to/your/datasets/{movies,fever,multirc} --output_dir /dir/to/your/trained_data --model_params ./params/movies_expred_lambda_5.json ```
