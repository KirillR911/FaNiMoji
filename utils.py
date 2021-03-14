import matplotlib
from PIL import ImageDraw, Image
from PIL import ImageFont
from matplotlib import pyplot as plt
import cv2
matplotlib.use('Agg')
import os
import yaml
from model import Prediction
import numpy as np
import torch

from sklearn.metrics import pairwise_distances


class_name = ("Blue", "Blue Gray", "Brown", "Brown Gray",
              "Brown Black", "Green", "Green Gray", "Other")
EyeColor = {
    class_name[0]: ((166, 21, 50), (240, 100, 85)),
    class_name[1]: ((166, 2, 25), (300, 20, 75)),
    class_name[2]: ((2, 20, 20), (40, 100, 60)),
    class_name[3]: ((20, 3, 30), (65, 60, 60)),
    class_name[4]: ((0, 10, 5), (40, 40, 25)),
    class_name[5]: ((60, 21, 50), (165, 100, 85)),
    class_name[6]: ((60, 2, 25), (165, 20, 65))
}

def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert(isinstance(size, torch.Size))
    return " × ".join(map(str, size))

def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print("%s:%s%s %s" % (type(obj).__name__,
                                          " GPU" if obj.is_cuda else "",
                                          " pinned" if obj.is_pinned else "",
                                          pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print("%s → %s:%s%s%s%s %s" % (type(obj).__name__,
                                                   type(obj.data).__name__,
                                                   " GPU" if obj.is_cuda else "",
                                                   " pinned" if obj.data.is_pinned else "",
                                                   " grad" if obj.requires_grad else "",
                                                   " volatile" if obj.volatile else "",
                                                   pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    print("Total size:", total_size)

def dist_bw_2(e1,e2):
    e1 = e1.reshape(1, -1)
    e2 = e2.reshape(1, -1)
    sim = 1 - pairwise_distances(e1,e2, metric = 'cosine')
    return sim[0][0]

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    if title is not None:
        ax.title.set_text(title)

    return ax



def get_batch(samples: list, device, size: int = 32) -> torch.Tensor:
    """
        Create batch from given data to pass to forward func.
    """
    while (len(samples) < size):
        samples.append(torch.zeros((3, 256, 256)))
    samples = torch.cat(samples)
    samples = samples.view((32, 3, 256, 256)).to(device)
    return samples


def get_size(img):
    if isinstance(img, np.ndarray):
        return img.shape[1::-1]
    else:
        return img.size

def dist_in_list(embs, target):
    sim = 1 - pairwise_distances(embs,target.reshape(1, -1), metric = 'cosine')
    sim.resize((1, sim.size))
    return sim[0]

def find_face_on_image(faces, target, image, title = "1.jpg"):

    img_draw = ImageDraw.Draw(image)
    dists = dist_in_list([face.getVector() for face in faces], target)
    face_boxes = [face.box_m for face in faces]
    ans = [0 for _ in range(len(dists))]
    font = ImageFont.truetype("./extra/arial_bold_italic.ttf", 8)
    ans[np.argmax(dists)] = 1 if max(dists) >= 0.6 else 0
    for i in range(len(faces)):
        img_draw.rectangle(face_boxes[i], width=4,
                           outline='red' if ans[i] == 0 else 'green')
        text = str(dists[i])
        text_w, text_h = img_draw.textsize(text, font)
        img_draw.text(((face_boxes[i][2]) - text_w, face_boxes[i][3]), text,
                      (255, 0, 0) if ans[i] == 0 else (128, 255, 0), font=font)
    if 1 not in ans:
        image.save(title.replace(".mp4", "FUCK.jpg"))
    return faces[ans.index(1)] if 1 in ans else None


def extract_faces(img_p, fa, saving = False,  margin=0) -> list:
    img = Image.fromarray(img_p)
    raw_image_size = get_size(img)

    boxes = fa.face_detector.detect_from_image(img_p)
    if boxes is None:
        return None, None, None
    k = 0
    imgs = []
    margd_boxes = []
    for idx, i in enumerate(boxes):
        i = i[:4]
        image_size = max(i[2] - i[0], i[3] - i[1])
        margin_t = [
            margin * (i[2] - i[0])/ 100,
            margin * (i[3] - i[1])/ 100,
        ]
        i_t = [
            int(max(i[0] - margin_t[0] / 2, 0)),
            int(max(i[1] - margin_t[1] / 2, 0)),
            int(min(i[2] + margin_t[0] / 2, raw_image_size[0])),
            int(min(i[3] + margin_t[1] / 2, raw_image_size[1])),
        ]
        margd_boxes.append(i_t)
        imgs.append(img.crop(i_t))
        if saving:
            if not os.path.exists("./tmp/"):
                os.makedirs("./tmp/")
            img.crop(i_t).save("./tmp/" + str(k) + ".jpg")
            k += 1
    return imgs, boxes, margd_boxes
def vis_landmark_on_img(shape, linewidth=2, img  = np.ones(shape=(256, 256, 3)) * 255):
    '''
    Visualize landmark on images.
    '''
    # img = np.ones(shape=(256, 256, 3)) * 255
    # shape += 256
    def draw_curve(idx_list, color=(0, 255, 0), loop=False, lineWidth=linewidth):
        for i in idx_list:
            cv2.line(img, (int(shape[i, 0]), int(shape[i, 1])), (int(shape[i + 1, 0]), int(shape[i + 1, 1])), color, lineWidth)
        if (loop):
            cv2.line(img, (int(shape[idx_list[0], 0]), int(shape[idx_list[0], 1])),
                     (int(shape[idx_list[-1] + 1, 0]), int(shape[idx_list[-1] + 1, 1])), color, lineWidth)
    draw_curve(list(range(0, 16)), color=(255, 144, 25))
    draw_curve(list(range(17, 21)), color=(50, 205, 50))
    draw_curve(list(range(22, 26)), color=(50, 205, 50))
    draw_curve(list(range(27, 35)), color=(208, 224, 63))
    draw_curve(list(range(36, 41)), loop=True, color=(71, 99, 255))
    draw_curve(list(range(42, 47)), loop=True, color=(71, 99, 255))
    draw_curve(list(range(48, 59)), loop=True, color=(238, 130, 238))
    draw_curve(list(range(60, 67)), loop=True, color=(238, 130, 238))
    left_eye = np.mean(shape[36:42], axis=0)
    right_eye = np.mean(shape[42:48], axis=0)
    left_eye = (left_eye[0], left_eye[1])
    right_eye = (right_eye[0], right_eye[1])
    eye_distance = np.linalg.norm(np.array(left_eye)-np.array(right_eye))
    eye_radius = eye_distance/15
    cv2.circle(img, left_eye, int(eye_radius), (0, 155, 255), 3)
    cv2.circle(img, right_eye, int(eye_radius), (0, 155, 255), 3)
    return img