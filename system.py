import torch
import shutil
from PIL import Image
################ Register
from mmcv import Config
from mmdet.apis import set_random_seed
from mmocr.datasets import build_dataset
from mmocr.models import build_detector
from mmocr.apis import train_detector
import os.path as osp
import mmcv
import pandas as pd
import matplotlib.pyplot as plt
from mmocr.apis import init_detector, model_inference

################ Province
import os
import cv2 as cv
import numpy as np
import random as rn
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import applications
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D,Input
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
############### Time
import time
############### Classification Report
from sklearn.metrics import classification_report, confusion_matrix

path_input = 'test/'
path_output_lp = 'out/license_plate/'
path_output_re = 'out/register/'
path_output_pro = 'out/province/'
for d in os.listdir(path_output_lp):
    os.remove(path_output_lp+d)
for d in os.listdir(path_output_re):
    os.remove(path_output_re+d)
for d in os.listdir(path_output_pro):
    os.remove(path_output_pro+d)
    
df = pd.read_csv('label.csv')
results = {'filenames': [],
           'actual register': [],
           'actual province' : [],
           'predict register': [],
           'predict province' : [],
          }
minmaxlp = {'filenames': [],
           'xmin': [],
           'ymin' : [],
           'xmax': [],
           'ymax' : [],
          }
minmaxre = {'filenames': [],
           'xmin': [],
           'ymin' : [],
           'xmax': [],
           'ymax' : [],
          }
minmaxpro = {'filenames': [],
           'xmin': [],
           'ymin' : [],
           'xmax': [],
           'ymax' : [],
          }
imgcontent = os.listdir(path_input)
imgcontent.sort()
time_list = []
for nameimg in imgcontent:
    start = time.time()
#######################license plate extraction

    # Model
    model = torch.hub.load('yolov5-master', 'custom' , path='Yolov5_Model_S/weights/best.pt' , source = 'local')

    # Images
    img = path_input+nameimg  # or file, Path, PIL, OpenCV, numpy, list

    # Inference
    resultyolo = model(img)

    # Results
    resultyolo.print()  # .print(), .show(), .save(), .crop(), .pandas(), etc.
    labels_detect_pill = resultyolo.xyxyn[0][:,5]
    list_pred = labels_detect_pill.tolist()
    data_result = resultyolo.pandas().xyxy[0]

    posit = 0
    position_lp = []
    position_re = []
    position_pro = []
    for a in list_pred:
        if a == 0.0:
            position_lp.append(posit)
        elif a == 1.0:
            position_re.append(posit)
        elif a == 2.0:
            position_pro.append(posit)
        posit += 1
    minmaxlp['filenames'].append(nameimg)
    minmaxre['filenames'].append(nameimg)
    minmaxpro['filenames'].append(nameimg)

    ### Select image
    if len(position_lp) > 1:      # เจอป้ายทะเบียน มากกว่า 1
        X_axis_distance = []
#         Y_axis_distance = []
        for lp in range(len(position_lp)):
            Xmin = data_result['xmin'][position_lp[lp]]
#             Ymin = data_result['ymin'][position_lp[lp]]
            Xmax = data_result['xmax'][position_lp[lp]]
#             Ymax = data_result['ymax'][position_lp[lp]]
            X_axis_distance.append(Xmax - Xmin)
#             Y_axis_distance.append(Ymax - Ymin)
        max_lp = X_axis_distance[0]  # list of X_axis_distance
        imax_lp = 0            # check position in list
        posi = 0                # sequence in list start from 0
        ##### check the most value of axis_distance
        for dis in X_axis_distance: 
            if (dis > max_lp): # check new value more than or less than old value
                max_lp = dis   # update parameter max value
                imax_lp = posi  # update position max value
            posi += 1   
        ## Xmin Ymin Xmax Ymax of lp
        xminlp = data_result['xmin'][position_lp[imax_lp]]
        yminlp = data_result['ymin'][position_lp[imax_lp]]
        xmaxlp = data_result['xmax'][position_lp[imax_lp]]
        ymaxlp = data_result['ymax'][position_lp[imax_lp]]
        minmaxlp['xmin'].append(xminlp)
        minmaxlp['ymin'].append(yminlp)
        minmaxlp['xmax'].append(xmaxlp)
        minmaxlp['ymax'].append(ymaxlp)
        img = Image.open(path_input+nameimg)
        box = (xminlp, yminlp, xmaxlp, ymaxlp)
        img2 = img.crop(box)
        img2.save(path_output_lp+nameimg)
    if len(position_lp) == 1:       # เจอป้ายทะเบียน 1 ป้าย
        ## Xmin Ymin Xmax Ymax of lp
        xminlp = data_result['xmin'][position_lp[0]]
        yminlp = data_result['ymin'][position_lp[0]]
        xmaxlp = data_result['xmax'][position_lp[0]]
        ymaxlp = data_result['ymax'][position_lp[0]]
        minmaxlp['xmin'].append(xminlp)
        minmaxlp['ymin'].append(yminlp)
        minmaxlp['xmax'].append(xmaxlp)
        minmaxlp['ymax'].append(ymaxlp)
        img = Image.open(path_input+nameimg)
        box = (xminlp, yminlp, xmaxlp, ymaxlp)
        img2 = img.crop(box)
        img2.save(path_output_lp+nameimg)
    if len(position_lp) == 0:    # ไม่เจอป้ายทะเบียน
        license_plate = 'not found license plate'

    ### Check conf of bbox
    if len(position_re) > 1 :      # register more than one
        conf_list = []     # add all confident
        count_posi_re = 0
        for re in position_re:  #1.upleft 2.upright 3.downright 4.downleft
            Xmin = data_result['xmin'][position_re[count_posi_re]]
            Ymin = data_result['ymin'][position_re[count_posi_re]]
            Xmax = data_result['xmax'][position_re[count_posi_re]]
            Ymax = data_result['ymax'][position_re[count_posi_re]]
            cenX = ((2*Xmin)+(2*Xmax))/4
            cenY = ((2*Ymin)+(2*Ymax))/4
            if xminlp<cenX<xmaxlp and yminlp<cenY<ymaxlp:
                conf_list.append(data_result['confidence'][re])
            count_posi_re += 1
        max_re = conf_list[0]  # max confidence value by start in first position
        imax_re = 0            # check position in list
        posi = 0                # sequence in list start from 0
        ##### check the most value of confi province
        for re1 in conf_list: 
            if (re1 > max_re): # check new value more than or less than old value
                max_re = re1   # update parameter max value
                imax_re = posi  # update position max value
            posi += 1           # update sequence
        xmin = data_result['xmin'][position_re[imax_re]]
        ymin = data_result['ymin'][position_re[imax_re]]
        xmax = data_result['xmax'][position_re[imax_re]]
        ymax = data_result['ymax'][position_re[imax_re]]
        minmaxre['xmin'].append(xmin)
        minmaxre['ymin'].append(ymin)
        minmaxre['xmax'].append(xmax)
        minmaxre['ymax'].append(ymax)
        img = Image.open(path_input+nameimg)
        box = (xmin, ymin, xmax, ymax)
        img2 = img.crop(box)
        img2.save(path_output_re+nameimg)
    if len(position_re) == 1 :
        conf_list = []     # add all confident
        for re in position_re:
            conf_list.append(data_result['confidence'][re])
        xmin = data_result['xmin'][position_re[0]]
        ymin = data_result['ymin'][position_re[0]]
        xmax = data_result['xmax'][position_re[0]]
        ymax = data_result['ymax'][position_re[0]]
        minmaxre['xmin'].append(xmin)
        minmaxre['ymin'].append(ymin)
        minmaxre['xmax'].append(xmax)
        minmaxre['ymax'].append(ymax)
        img = Image.open(path_input+nameimg)
        box = (xmin, ymin, xmax, ymax)
        img2 = img.crop(box)
        img2.save(path_output_re+nameimg)
    if len(position_re) == 0:
        register = 'not found license register'
    if len(position_pro) > 1 :     # province more than one
        conf_list = []     # add all confident
        count_posi_pro = 0
        for pro in position_pro:  #1.upleft 2.upright 3.downright 4.downleft
            Xmin = data_result['xmin'][position_pro[count_posi_pro]]
            Ymin = data_result['ymin'][position_pro[count_posi_pro]]
            Xmax = data_result['xmax'][position_pro[count_posi_pro]]
            Ymax = data_result['ymax'][position_pro[count_posi_pro]]
            cenX = ((2*Xmin)+(2*Xmax))/4
            cenY = ((2*Ymin)+(2*Ymax))/4
            if xminlp<cenX<xmaxlp and yminlp<cenY<ymaxlp:
                conf_list.append(data_result['confidence'][pro])
            count_posi_pro += 1
        max_pro = conf_list[0]  # max confidence value by start in first position
        imax_pro = 0            # check position in list
        posi = 0                # sequence in list start from 0
        ##### check the most value of confi province
        for pro1 in conf_list: 
            if (pro1 > max_pro): # check new value more than or less than old value
                max_pro = pro1   # update parameter max value
                imax_pro = posi  # update position max value
            posi += 1           # update sequence
        xmin = data_result['xmin'][position_pro[imax_pro]]
        ymin = data_result['ymin'][position_pro[imax_pro]]
        xmax = data_result['xmax'][position_pro[imax_pro]]
        ymax = data_result['ymax'][position_pro[imax_pro]]
        minmaxpro['xmin'].append(xmin)
        minmaxpro['ymin'].append(ymin)
        minmaxpro['xmax'].append(xmax)
        minmaxpro['ymax'].append(ymax)
        img = Image.open(path_input+nameimg)
        box = (xmin, ymin, xmax, ymax)
        img2 = img.crop(box)
        img2.save(path_output_pro+nameimg)
    if len(position_pro) == 1 :
        conf_list = []     # add all confident 
        for pro in position_pro:
            conf_list.append(data_result['confidence'][pro])
        xmin = data_result['xmin'][position_pro[0]]
        ymin = data_result['ymin'][position_pro[0]]
        xmax = data_result['xmax'][position_pro[0]]
        ymax = data_result['ymax'][position_pro[0]]
        minmaxpro['xmin'].append(xmin)
        minmaxpro['ymin'].append(ymin)
        minmaxpro['xmax'].append(xmax)
        minmaxpro['ymax'].append(ymax)
        img = Image.open(path_input+nameimg)
        box = (xmin, ymin, xmax, ymax)
        img2 = img.crop(box)
        img2.save(path_output_pro+nameimg)
    if len(position_pro) == 0:
        province = 'not found license province'

#######################register

    cfg = Config.fromfile('mmocr-main/configs/textrecog/sar/sar_r31_parallel_decoder_lp_data.py')

    cfg.work_dir = 'mmocr-main/demo/tutorial_exps'

    cfg.optimizer.lr = 0.001 / 8
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 40

    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    datasets = [build_dataset(cfg.data.train)]
    torch.cuda.set_device(2)

    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

    model.CLASSES = datasets[0].CLASSES

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    img = path_output_re+nameimg
    checkpoint = "mmocr-main/demo/tutorial_exps4/epoch_5.pth"

    model = init_detector(cfg, checkpoint, device="cuda:0")
    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][0].pipeline


    result = model_inference(model, img)



######################province
    list_province = ['กระบี่', 'กรุงเทพมหานคร', 'กาญจนบุรี', 'กาฬสินธุ์', 'กำแพงเพชร', 'ขอนแก่น', 'จันทบุรี', 'ฉะเชิงเทรา', 'ชลบุรี', 'ชัยนาท', 'ชัยภูมิ', 'ชุมพร', 'ตรัง', 'ตราด', 'ตาก', 'นครนายก', 'นครปฐม', 'นครพนม', 'นครราชสีมา', 'นครศรีธรรมราช', 'นครสวรรค์', 'นนทบุรี', 'นราธิวาส', 'น่าน', 'บึงกาฬ', 'บุรีรัมย์', 'ปทุมธานี', 'ประจวบคีรีขันธ์', 'ปราจีนบุรี', 'ปัตตานี', 'พระนครศรีอยุธยา', 'พะเยา', 'พังงา', 'พัทลุง', 'พิจิตร', 'พิษณุโลก', 'ภูเก็ต', 'มหาสารคาม', 'มุกดาหาร', 'ยะลา', 'ยโสธร', 'ระนอง', 'ระยอง', 'ราชบุรี', 'ร้อยเอ็ด', 'ลพบุรี', 'ลำปาง', 'ลำพูน', 'ศรีสะเกษ', 'สกลนคร', 'สงขลา', 'สตูล', 'สมุทรปราการ', 'สมุทรสงคราม', 'สมุทรสาคร', 'สระบุรี', 'สระแก้ว', 'สิงห์บุรี', 'สุพรรณบุรี', 'สุราษฎร์ธานี', 'สุรินทร์', 'สุโขทัย', 'หนองคาย', 'หนองบัวลำภู', 'อำนาจเจริญ', 'อุดรธานี', 'อุตรดิตถ์', 'อุทัยธานี', 'อุบลราชธานี', 'อ่างทอง', 'เชียงราย', 'เชียงใหม่', 'เบตง', 'เพชรบุรี', 'เพชรบูรณ์', 'เลย', 'เเพร่', 'แม่ฮ่องสอน']

    def non_trainable(model):
        for i in range(len(model.layers)):
            model.layers[i].trainable = False
        return model

    def create_model(n_classes,output_activation):
        os.environ['PYTHONHASHSEED'] = '0'
        tf.keras.backend.clear_session()

        ## Set the random seed values to regenerate the model.
        np.random.seed(0)
        rn.seed(0)

        #Input layer
        input_layer = Input(shape=(256,256,3),name='Input_Layer')

        #Adding pretrained model
        resnet = applications.ResNet50(include_top=False,weights='imagenet',input_tensor = input_layer)

        #Flatten
        flatten = Flatten(data_format='channels_last',name='Flatten')(resnet.output)

        #FC layer
        FC1 = Dense(units=512,activation='relu',name='FC1')(flatten)

        #FC layer
        FC2 = Dense(units=256,activation='relu',name='FC2')(FC1)

        #Dropout layer
        droput1 = Dropout(0.5)(FC2)

        #output layer
        Out = Dense(units=n_classes,activation=output_activation,name='Output')(droput1)

        #Creating the Model
        model = Model(inputs=input_layer,outputs=Out)

        return model


    with tf.device('/device:GPU:0'):
        resnet = applications.ResNet50(include_top=False,weights='imagenet',input_shape=(256,256,3))
        resnet = non_trainable(resnet)
        fc = Flatten()(resnet.output)
        model_resnet = Model(inputs = resnet.input,outputs = fc)
        model = create_model(78,'softmax')
        model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.SGD(lr=0.00001, momentum=0.9), metrics=["accuracy"])

        model.load_weights(
                "province_classification/resnet-0.951.hdf5"
            )

        img_path = path_output_pro+nameimg
        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img)

        img_batch = np.expand_dims(img_array, axis=0)

        img_preprocessed = preprocess_input(img_batch)

        prediction = model.predict(img_preprocessed)

        y_pred = np.argmax(prediction,axis=1)
    
    end = time.time() - start
    time_list.append(end)
    results['filenames'].append(nameimg)
    results['actual register'].append(df['เลขทะเบียน'][df['filenames'] == nameimg].tolist()[0])
    results['actual province'].append(df['จังหวัด'][df['filenames'] == nameimg].tolist()[0])
    results['predict register'].append(result['text'])
    results['predict province'].append(list_province[int(y_pred)])
    
results_df = pd.DataFrame.from_dict(results)
minmaxlp_df = pd.DataFrame.from_dict(minmaxlp)
minmaxre_df = pd.DataFrame.from_dict(minmaxre)
minmaxpro_df = pd.DataFrame.from_dict(minmaxpro)

results_df['actual register'] = results_df['actual register'].apply(str)
results_df['predict register'] = results_df['predict register'].apply(str)
y_pred = results_df['actual register']
y_true = results_df['predict register']
print('Exact match register :',accuracy_score(y_true, y_pred))

results_df['actual province'] = results_df['actual province'].apply(str)
results_df['predict province'] = results_df['predict province'].apply(str)
y_pred = results_df['actual province']
y_true = results_df['predict province']
print('Exact match province :',accuracy_score(y_true, y_pred))

def occurrences(y_true, y_pred):
    count = 0
    y_true_ = [char for char in y_true]
    y_pred_ = [char for char in y_pred]
    missed_pred = []
    for char in y_pred_:
        if char in y_true_:
            count += 1
            y_true_.remove(char)
        else:
            missed_pred.append(char)
    missed_y_true = y_true_
    return count/len(y_true), missed_pred, missed_y_true

scores, missed_preds, missed_y_trues = [], [], []
for i in range(len(results_df['actual register'])):
    score, missed_pred, missed_y_true = occurrences(results_df['actual register'][i], results_df['predict register'][i])
    scores.append(score)
    missed_preds.append(missed_pred)
    missed_y_trues.append(missed_y_true)
print('Character register :',sum(scores)/len(results_df['actual register']))

scores, missed_preds, missed_y_trues = [], [], []
for i in range(len(results_df['actual province'])):
    score, missed_pred, missed_y_true = occurrences(results_df['actual province'][i], results_df['predict province'][i])
    scores.append(score)
    missed_preds.append(missed_pred)
    missed_y_trues.append(missed_y_true)
print('Character province :',sum(scores)/len(results_df['actual province']))

sumtime = 0
for t in range(len(time_list)):
    sumtime = sumtime + time_list[t]

average_time = sumtime/len(time_list)
print('average_time :',average_time)

results_df.to_excel('system.xlsx')
results_df.to_csv('system.csv')
minmaxlp_df.to_csv('minmaxlp1.csv')
minmaxre_df.to_csv('minmaxre1.csv')
minmaxpro_df.to_csv('minmaxpro1.csv')











