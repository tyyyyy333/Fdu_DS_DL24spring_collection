HERE YOU CAN TARIN THE CIFAR-10 DATASET AND TEST THEM.
run the code at command line like these:
- python train.py --model=CANet --model_path=cifar10/saved_model/model.pth --data_path=../data/ --log_path=runs/CANet
- python visualization.py --model=CANet --model_path=saved_model/CANet.pth  --log_path=runs/CANet
- python test.py --model=CANet --model_path=saved_model/CANet.pth
- python model.py --model=CANet_pro
MAKE SURE YOU HAVE DOWNLOAD THE CORRESPONDING MODEL WEIGHTS OR LOGS BEFORE YOU RUN `test.py` OR `visualization.py`
