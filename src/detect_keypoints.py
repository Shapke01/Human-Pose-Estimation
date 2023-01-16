import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn
import cv2
import numpy as np

root_folder = Path(__file__).parents[1]
img_path = root_folder / "data/mpii_human_pose_v1/images/"

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


img = cv2.imread(str(img_path.joinpath('000001163.jpg')))

# preprocess the input image
transform = T.Compose([T.ToTensor()])
img_tensor = transform(img)

# forward-pass the model
# the input is a list, hence the output will also be a list
output = model([img_tensor])[0]

def draw_keypoints_per_person(img, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
    # initialize a set of colors from the rainbow spectrum
    cmap = plt.get_cmap('rainbow')
    # create a copy of the image
    img_copy = img.copy()
    # pick a set of N color-ids from the spectrum
    color_id = np.arange(1,255, 255//len(all_keypoints)).tolist()[::-1]
    # iterate for every person detected
    for person_id in range(len(all_keypoints)):
        # check the confidence score of the detected person
        if confs[person_id]>conf_threshold:
        # grab the keypoint-locations for the detected person
            keypoints = all_keypoints[person_id, ...]
            # grab the keypoint-scores for the keypoints
            scores = all_scores[person_id, ...]
            # iterate for every keypoint-score
            for kp in range(len(scores)):
                # check the confidence score of detected keypoint
                if scores[kp]>keypoint_threshold:
                    # convert the keypoint float-array to a python-list of integers
                    keypoint = tuple(map(int, keypoints[kp, :2].detach().numpy().tolist()))
                    # pick the color at the specific color-id
                    color = tuple(np.asarray(cmap(color_id[person_id])[:-1])*255)
                    # draw a circle over the keypoint location
                    cv2.circle(img_copy, keypoint, 5, color, -1)

    return img_copy

keypoints_img = draw_keypoints_per_person(img, output["keypoints"], output["keypoints_scores"], output["scores"], keypoint_threshold=2)

cv2.imshow('Sample', keypoints_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
