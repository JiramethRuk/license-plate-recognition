ในไฟล์ system จะเป็นไฟล์ที่รวมระบบของโปรเจกต์ อ่านป้ายทะเบียน

Dataset 
data ที่ใช้ train YOLOv5 
- data_yolo : รูปภาพรถเต็มคัน
data ที่ใช้ train text recognition
- data_re : รูปภาพรถเต็มคัน
- data_re1 : รูปภาพเลขะเบียนจริงๆ
- data_re2 : รูป synthetic data แบบ เพิ่มความหลากหลายของสีฟอนต์ ไม่มี noise
- data_re3 : รูป synthetic data แบบ เพิ่มความหลากหลายของสีฟอนต์ มี noise
data ที่ใช้ train province classification
- data_real : รูปภาพจังหวัดจริงๆ
- data_synv1 : รูป synthetic data แบบ สีฟอนต์เป็นสีดำ ไม่มี noise
- data_synv2 : รูป synthetic data แบบ สีฟอนต์เป็นสีดำ มี noise
- data_synv3 : รูป synthetic data แบบ เพิ่มความหลากหลายของสีฟอนต์ ไม่มี noise
- data_synv4 : รูป synthetic data แบบ เพิ่มความหลากหลายของสีฟอนต์ มี noise

ระบบจะได้รับ input มาเป็นรูปรถเต็มค้น ซึ้งสามารถใส่รูปภาพไว้ที่ไฟล์ test ได้เลย ซึ่งสามารถใส่กี่รูปก็ได้
มี label ของ test อยู่ในไฟล์ label_test

ในการทำงานของระบบจะมีทั้งหมดอยู่ 3 ส่วน

ส่วนที่ 1 : ตรวจจับป้ายทะเบียน 
ส่วนนี้จะมีไว้เพื่อระบุตำแหน่งของป้ายทะเบียนว่าอยู่ส่วนไหนของรูปภาพ โดยจะใช้เป็น Object Detection ของ YOLOv5 (Model S)
github : https://github.com/ultralytics/yolov5 (git clone มาแล้ว สามารถลองเทรนโมเดลอื่นได้)
git clone : yolov5-master
path weight ที่เทรนแล้ว : https://kasikornbankgroup.sharepoint.com/:f:/s/OCR/En1SPrzhSVpHnBqcYzo280kBetgdunFLQ8GGPt3YyhGagQ?e=vSyrru
ขันตอนในการทำงาน
- โหลด weight จากไฟล์ Yolov5_Model_S
- ระบบจะได้รับรูปมาจากไฟล์ test เป็นรูปรถเต็มคัน
- object detection จะตรวจจับทั้งหมด 3 อย่าง ได้แก่ 1.ป้ายทะเบียน 2.เลขทะเบียน 3.จังหวัด
- use case ที่ 1 : รูปภาพมีป้ายทะเบียนมากกว่า 1 ป้าย
    ถ้าตรวจจับเจอป้ายทะเบียนมากกว่า 1 ป้าย ระบบจะทำการเลือกป้ายทะเบียน เพราะ ระบบจะอ่านป้ายทะเบียน 1 ป้าย ต่อ 1 รูป เท่านั้น
    ระบบจะทำการเช็ค ความกว้างของป้ายทะเบียนทั้งหมด และเลือกใช้ ป้ายทะเบียนเพียงป้ายเดียว โดยป้ายทะเบียนที่เลือกมาจะมีการเก็บตำแหน่งของ Xmin Ymin Xmax Ymax ของ ป้ายทะเบียนด้วย
- use case ที่ 2 : เลือกเลขทะเบียนกับจังหวัดของป้ายทะเบียน
    ทำการคัดเลือกเลขทะเบียนหรือจังหวัดที่อยู่ภายในป้ายทะเบียนที่กำหนกว้
    ถ้าเกิด error ในการตรวจจับ เลขทะเบียนหรือจังหวัดที่อยู่ภายในป้ายทะเบียนที่กำหนดไว้จะทำการเช็คค่า confident ที่ได้จาก object detection และทำการเลือกการเพียงอันเดียวเท่านั้น
- ผลที่ได้จากระบบจะทำการจัดเก็บรูปภาพไว้ที่ไฟล์ out โดยภายในจะประกอบด้วยไฟล์ licence plate จัดเก็บรูปป้ายทะเบียน, ไฟล์ Register จัดเก็บรูปเลขทะเบียน, ไฟล์ Province จัดเก็บรูปจังหวัด

ส่วนที่ 2 : อ่านเลขทะเบียน
ส่วนนี้จะมีไว้เพื่ออ่านเลขทะเบียน โดยจะเป็นการอ่านเลขทะเบียนโดยใช้ SAR Model (Text Recognition Model)
github : https://github.com/open-mmlab/mmocr/tree/5fc920495acf3e1e1c933f85b284f7384481bf71 
(git clone มาแล้ว สามารถลองเทรนโมเดลอื่นได้)
git clone : mmocr-main
path weight ที่เทรนแล้ว : https://kasikornbankgroup.sharepoint.com/:f:/s/OCR/Esciz2fePx5Hp1R1X_xbKxcBjnMKBO4Vv3h4kuPMJc9-CA?e=bZTZOn
data for train:
image : รูปภาพเป็นรูปที่ถูกตัดให้เหลือเลขทะเบียนแล้ว
label : เป็นไฟล์ txt ที่ระบุชื่อไฟล์ของรูปภาพและข้อความ เช่น '1.jpg กข1234'
สามารถศึกษาเพิ่มเติมได้ที่เกี่ยว dataset ได้ที่ : https://mmocr.readthedocs.io/en/latest/tutorials/dataset_types.html#text-recognition
ศึกษาเกี่ยวกับไฟล์ config file : https://mmocr.readthedocs.io/en/latest/tutorials/config.html
ขั้นตอนในการทำงาน
- โหลดไฟล์ Confi ที่จัดเตรียมไว้
- โหลด weight ที่จะมาอ่านเลขทะเบียน
- นำรูปมาจากไฟล์ Register

ส่วนที่ 3 : อ่านจังหวัด
ส่วนนี้จะมีไว้เพื่ออ่านจังหวัด โดยจะเป็นการอ่านจังหวัดในป้ายทะเบียนโดยใช้ ResNet50 (Classification Model)
path weight ที่เทรนแล้ว : https://kasikornbankgroup.sharepoint.com/:f:/s/OCR/ErIXvs-Vcc5OiX2L5mAjpWIB6OM11XK8syNWk_1QJ_bhPw?e=AzOr0F
- โหลด pretrained model จาก tensorflow.keras
- โหลด weight ที่เทรนเอาไว้ อยู่ในไฟล์ province_classification
- นำรูปมาจากไฟล์ Province

ผลลัพธ์จะถูกจัดเก็บอยู่ในไฟล์ excel และไฟล์ csv