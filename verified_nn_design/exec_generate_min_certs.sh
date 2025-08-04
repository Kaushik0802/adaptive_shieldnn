#!/bin/bash
rm -r __pycache__
# source activate tensorflow2_p36
export PYTHONPATH="/home/ubuntu/src/cntk/bindings/python:/home/ubuntu/anaconda3/lib/python36.zip:/home/ubuntu/anaconda3/lib/python3.6:/home/ubuntu/anaconda3/lib/python3.6/lib-dynload:/home/ubuntu/anaconda3/lib/python3.6/site-packages"
python3 ./generate_min_certs.py > ./stdout.txt

sudo shutdown