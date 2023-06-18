import argparse
import os
import shutil
import time
from pathlib import Path
from PIL import Image
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from imgviz import rgb2gray
from matplotlib import pyplot as plt
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

ss = 0


def detect(save_img, source, weights, save_txt, imgsz, nosave, project, name, exist_ok, device, augment,
           conf_thres, iou_thres, classes, agnostic_nms, save_conf, view_img, flag):
    sss1 = ""  # 银行名字
    sss2 = ""  # 卡的类型
    sss3 = ""  # 银行卡号
    global ss
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # 目录
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 如果要保存框坐标txt文件就创建一个新文件保存

    # 初始化
    set_logging()  # 调用设置日志记录，记录程序运行的时间、输出的结果等
    device = select_device(device)  # 选择使用设备
    half = device.type != 'cpu'  # 判断是否支持半精度 仅CUDA支持半精度

    # 荷载模型
    model = attempt_load(weights, map_location=device)  # 利用权重加载模型
    stride = int(model.stride.max())  # 获取模型的步长
    imgsz = check_img_size(imgsz, s=stride)  # 利用步长检查输入图片的大小是否符合要求
    if half:
        model.half()  # 将模型转换为FP16格式

    # 第二级分类器
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # 初始化
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # 数据加载器
    vid_path, vid_writer = None, None
    if webcam:  # 判断是否使用摄像头
        view_img = check_imshow()
        cudnn.benchmark = True  # 设置为True以加快恒定图像大小推断
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True  # 需要保存推理结果
        cudnn.benchmark = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # 获取名称和颜色
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # 运行推理
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:  # path表示图像的路径，img表示图像的numpy数组，im0s表示原始图像的numpy数组，vid_cap表示视频捕获对象

        imm = im0s

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 转化为fp16/32格式
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推断
        t1 = time_synchronized()  # 记录当前时间
        pred = model(img, augment=augment)[0]  # 利用加载的模型进行推断

        # 应用nms
        # 对预测结果进行非极大值抑制，得到最终的检测结果
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        # 运用分类器
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)  # 检测结果进行分类

        # 过程检测
        for i, det in enumerate(pred):  # 获取预测内容
            if webcam:  # batch_size >= 1
                # 预测图片  输出字符串  原始图片副本
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            # 保存路径
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # 将框从img_size重新缩放为im0大小
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 打印结果
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到字符串

                # 记录结果
                ss = ss + 1

                xx = []
                yy = []
                for *xyxy, conf, cls in reversed(det):  # 获取每个检测框的坐标、置信度和类别信息
                    # 并将坐标信息解包成4个变量 左上右下坐标
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

                    if flag == 1:  # 第二次预测
                        xx.append(c1[0])  # 记录左上坐标
                        yy.append(int(cls))  # 同时记录数字号码

                    if flag == 0:  # 第一次预测
                        if 0 <= int(cls) <= 6:
                            if int(cls) == 0:
                                sss1 = "招商银行"
                            if int(cls) == 1:
                                sss1 = "交通银行"
                            if int(cls) == 2:
                                sss1 = "中国民生银行"
                            if int(cls) == 3:
                                sss1 = "中国银行"
                            if int(cls) == 4:
                                sss1 = "中国工商银行"
                            if int(cls) == 5:
                                sss1 = "中国建设银行"
                            if int(cls) == 6:
                                sss1 = "中国农业银行"
                        if 7 <= int(cls) <= 8:
                            if int(cls) == 7:
                                sss2 = "借记卡"
                            if int(cls) == 8:
                                sss2 = "信用卡"

                    if flag == 0:  # 第一次预测
                        if int(cls) == 9:  # 把银行卡号码图片截取保存用于进行第二次卡号预测
                            pic = imm[c1[1]:c2[1], c1[0]:c2[0]]
                            pic = rgb2gray(pic)
                            zz = "E:/YOLOv5/hhe+12003990219/bank_number/image"

                            shutil.rmtree(zz)
                            os.mkdir(zz)

                            path_1 = zz + "/ss" + str(ss) + ".jpg"
                            cv2.imwrite(path_1, pic)

                    if save_txt:  # 如果要保存txt文件就写入
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    if flag == 1:  # 第二次号码预测
                        total_circle = len(xx) - 1
                        # 冒泡排序 利用xx存储每个数字外框的左上坐标大小顺序对yy存储的数字进行排序 得到有序的卡号
                        for z in range(total_circle):  # 外层循环，循环n-1次
                            index = 0
                            flag1 = 0  # 0表示没有交换过
                            while index <= total_circle - 1:  # 内层循环，每轮循环期间，需要两两比较(n-1)-1次
                                if xx[index] > xx[index + 1]:  # 比较两个数的大小，左边xx大则交换
                                    xx[index], xx[index + 1] = xx[index + 1], xx[index]
                                    yy[index], yy[index + 1] = yy[index + 1], yy[index]
                                    flag1 = 1
                                index = index + 1  # 从左往右依次比较两个数的大小，索引+1进行移动
                            if flag1 == 0:  # 当在一次找最大值的过程中，未交换过两个数，则直接退出排序过程，即退出外层循环。
                                break
                            total_circle = total_circle - 1
                if flag == 1:  # 第二次卡号预测，记录排序后的卡号
                    tt = 0
                    for indx in yy:
                        sss3 += str(indx)
                        tt += 1
                        if tt % 4 == 0:
                            sss3 += '   '

            # 打印时间
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # 根据参数展示预测后的结果
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            # 保存预测结果
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)

    print(f'Done. ({time.time() - t0:.3f}s)')
    return sss1, sss2, sss3


def fun1(source1):
    flag = 0
    # 训练权重路径
    weights = 'runs/train/exp/weights/best.pt'
    s = "alone"
    s1 = ""
    s2 = ""
    s3 = ""

    if not os.path.exists(s):
        os.mkdir(s)
    else:
        shutil.rmtree(s)
        os.mkdir(s)
    img = cv2.imread(source1)
    path_1 = s + "/ss.jpg"
    cv2.imwrite(path_1, img)

    source = s  # 测试数据
    img_size = 640  # 输入图片大小
    save_txt = False  # 是否将预测框坐标以txt保存
    nosave = False  # 不保存模型
    project = 'runs/detect'  # 推理结果存放路径
    name = 'exp'  # 结果保存的文件夹名称
    exist_ok = False  # 模型目录是否存在
    device = ''  # 选择使用设备 cpu；0(表示一个gpu设备cuda:0)；0,1,2,3(多个gpu设备)
    augment = False
    conf_thres = 0.25  # 置信度阈值
    iou_thres = 0.45  # 做nms的iou阈值
    agnostic_nms = False  # 进行nms是否也去除不同类别之间的框
    save_conf = False
    update = False  # 如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    save_img = False
    classes = None  # 设置只保留某一部分类别，形如0或者0 2 3
    view_img = False  # 是否展示预测之后的图片/视频，默认False

    with torch.no_grad():
        if update:  # update all models (to fix SourceChangeWarning)
            s1, s2, s3 = detect(save_img, source, weights, save_txt, img_size, nosave, project, name, exist_ok, device,
                                augment,
                                conf_thres, iou_thres, classes, agnostic_nms, save_conf, view_img, flag)
            strip_optimizer(weights)
        else:
            s1, s2, s3 = detect(save_img, source, weights, save_txt, img_size, nosave, project, name, exist_ok, device,
                                augment,
                                conf_thres, iou_thres, classes, agnostic_nms, save_conf, view_img, flag)
    return s1, s2, s3


def fun2():
    flag = 1
    s1 = ""
    s2 = ""
    s3 = ""
    weights = 'runs/train/exp6/weights/best.pt'
    source = 'bank_number/image'
    img_size = 640
    save_txt = False
    nosave = False
    project = 'runs/detect'
    name = 'exp'
    exist_ok = False
    device = ''
    augment = False
    conf_thres = 0.25
    iou_thres = 0.45
    agnostic_nms = False
    save_conf = False
    update = False
    save_img = False
    classes = None
    view_img = False

    with torch.no_grad():
        if update:  # update all models (to fix SourceChangeWarning)
            s1, s2, s3 = detect(save_img, source, weights, save_txt, img_size, nosave, project, name, exist_ok, device,
                                augment,
                                conf_thres, iou_thres, classes, agnostic_nms, save_conf, view_img, flag)
            strip_optimizer(weights)
        else:
            s1, s2, s3 = detect(save_img, source, weights, save_txt, img_size, nosave, project, name, exist_ok, device,
                                augment,
                                conf_thres, iou_thres, classes, agnostic_nms, save_conf, view_img, flag)
    return s1, s2, s3
