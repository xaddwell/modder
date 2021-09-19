import time
import win32gui, win32ui, win32con, win32api
from Mod.detection.MtcnnDetector import MtcnnDetector
from Mod.detection.detector import Detector
from Mod.detection.fcn_detector import FcnDetector
from Mod.model import P_Net,R_Net,O_Net
import cv2
import numpy as np
import Mod.config as config
from copy import copy

test_mode=config.test_mode
thresh=config.thresh
min_face_size=config.min_face
stride=config.stride
detectors=[None,None,None]
# 模型放置位置
model_path=['Mod/model/PNet/','Mod/model/RNet/','Mod/model/ONet']
batch_size=config.batches
PNet=FcnDetector(P_Net,model_path[0])
detectors[0]=PNet
kk=1.25
if test_mode in ["RNet", "ONet"]:
    RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
    detectors[1] = RNet
if test_mode == "ONet":
    ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
    detectors[2] = ONet
mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,stride=stride, threshold=thresh)
out_path=config.out_path
def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
def capLoc(img):
    boxes_c, landmarks = mtcnn_detector.detect(img)
    loc=[]
    # print(landmarks)
    image=img.copy()
    for i in range(landmarks.shape[0]):
        cv2.circle(image, (int(landmarks[i][4]), int(int(landmarks[i][5]))), 5, (0, 0, 255))
        loc.append([int(landmarks[i][4]), int(int(landmarks[i][5]))])
    # cv2.imshow('im', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return loc
def autoMouse(temp):
    x=int(temp[0]/kk)
    y=int(temp[1]/kk)
    win32api.SetCursorPos([x,y])
    time.sleep(0.5)
    # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
def clickPos(pos=None):
    pass
def main():
    while True:
        print("start")
        image_array= grab_screen(region=(0, 0, 1920, 1080))[:,:,::-1]
        # 获取屏幕，(0, 0, 1280, 720)表示从屏幕坐标（0,0）即左上角，截取往右1280和往下720的画面
        loc = capLoc(image_array)

        # print(loc)
        for temp in loc:
            autoMouse(temp)
            print(temp)
main()