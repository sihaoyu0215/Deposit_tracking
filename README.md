# Deposit_tracking
This project is for efficient and accurate deposit detection and tracking within pipelines, based on combination of YOLOX (for detection) and BYTE (for tracking).

The framework is developed based on the open-source tool MMTracking. The model configuration file used in this paper is located at "./configs/mot/myconfigs/deposit_tracking_bytetrack.py", with the trained parameter file shared in "./model.pth". To reproduce the training process in the paper, you need to first label your own data and convert it to the CocoVID format. Then run train.py in the "./tools" folder based on your own configuration file (or use the provided configuration file).

The video data annotation is conducted with the aid of Computer Vision Annotation Tool (CVAT).

# Citation
Citation link will be released upon paper acceptance.
