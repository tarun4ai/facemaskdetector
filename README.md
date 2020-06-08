# facemaskdetector
ONNX model for facemask detection
This code is based on https://github.com/chenjun2hao/facemask.
Significant speed improvement after converting to ONNX

Requirements:-
Create python environment with the following packages:-
1. onnxruntime
2. opencv
3. Pytorch
4. NumPy (should be installed with opencv automatically)
5. easydict

Test code is inside main.py
It is connecting to integrated camera using opencv (source 0) and doing inferencing on every frame.
Change video source to the desred video ccording to you need.
Or modify code to test on images.

contact: er.tarunmishra@gmail.com
