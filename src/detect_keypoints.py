import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn

root_folder = Path(__file__).parents[1]
img_path = root_folder / "data/mpii_human_pose_v1/images"

print(img_path)


# create a model object from the keypointrcnn_resnet50_fpn class
model = keypointrcnn_resnet50_fpn(pretrained=True)
# call the eval() method to prepare the model for inference mode.
model.eval()
# create the list of keypoints.
keypoints = ['nose','left_eye','right_eye',\
'left_ear','right_ear','left_shoulder',\
'right_shoulder','left_elbow','right_elbow',\
'left_wrist','right_wrist','left_hip',\
'right_hip','left_knee', 'right_knee', \
'left_ankle','right_ankle']

