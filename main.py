import argparse
import os
import re
import socket
import sys

import cv2
import numpy as np
import pyocr
import pyocr.builders
from flask import Flask, request
from numpy.compat import py3k
from numpy.core.fromnumeric import sort
from numpy.testing._private.utils import print_assert_equal
from PIL import Image

app = Flask(__name__)

class Plate:
    def __init__(self):
        self.BRAND = None
        self.COUNTRY = None
        self.TYPE = None
        self.NUMBER = None
        self.ENGINE = None # MODEL
        self.POWER = None
        self.GVM = None
        self.SIZE = None
        self.DATEy = None
        self.DATEm = None
        self.OCCUPANTS = None
        self.MANUFACTURER = None
        self.MANUFACTURERn2 = None

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

@app.route('/', methods=['POST'])
def main(args=None, tool=None):
    if args:
        ori_img = cv2.imread(args.sample_path)
        ori_img_l = ori_img.shape[0]
        ori_img_w = ori_img.shape[1]

        perspective_img = ori_img.copy()
        flag_write = args.write_image
    else: #Flask
        path_received_socket = 'received_socket/temp.png'
        request.files.get('file').save(path_received_socket)
        ori_img = cv2.imread(path_received_socket)
        perspective_img = ori_img.copy()
        flag_write = app.config.get('write_image')
        tool = app.config.get('tool')

    img = cv2.cvtColor(ori_img,cv2.COLOR_BGR2GRAY)
    if flag_write:
        cv2.imwrite('output/intermediate_image.png', img)

    ret, img = cv2.threshold(img,75,255,0)
    if flag_write:
        cv2.imwrite('output/intermediate_image.png', img)

    img = cv2.bitwise_not(img)
    if flag_write:
        cv2.imwrite('output/intermediate_image.png', img)

    # 白色区域变大
    kernel_dilate = np.ones((5,5),np.uint8)
    img = cv2.dilate(img,kernel_dilate,iterations = 3)
    if flag_write:
        cv2.imwrite('output/intermediate_image.png', img)

    # 白色区域变小
    kernel_erode = np.ones((5,5),np.uint8)
    img = cv2.erode(img,kernel_erode,iterations = 3)
    if flag_write:
        cv2.imwrite('output/intermediate_image.png', img)

    # 找到轮廓
    image, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    cnt = contours[1] # 需要循环操作，找到拍照区域
    # img = cv2.drawContours(ori_img, contours, -1, (0,255,0), 3)

    for cnt in contours:
        img = cv2.drawContours(ori_img, cnt, -1, (0,255,0), 9)
        if flag_write:
            cv2.imwrite('output/square_detector.png', img)

        # img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
        # cv2.imwrite('output/square_detector.png', img)

        # 轮廓的近似
        epsilon = 0.08*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        img = cv2.drawContours(img, approx, -1, (0,0,255), 15)
        if flag_write:
            cv2.imwrite('output/square_detector.png', img)

        if approx.shape[0] == 4:
            # cv2.warpPerspective
            # 会出现长宽颠倒的现象，需要解决。
            approx= np.squeeze(approx)

            # 取两个最小x的方法
            point_1 = approx[0]
            point_2 = approx[1]
            point_3 = approx[2]
            point_4 = approx[3]
            points = [point_1, point_2, point_3, point_4]

            points = sorted(points, key=lambda x: x[0])
            points_left = points[0:2]
            points_right = points[2:4]

            points_left = sorted(points_left, key=lambda x: x[1])
            points_right = sorted(points_right, key=lambda x: x[1])

            # p1 #左上
            # p4 #右下
            # p3 #左下
            # p2 #右上
            p1 = points_left[0]
            p3 = points_left[1]

            p2 = points_right[0]
            p4 = points_right[1]

            o_width = np.linalg.norm(p2 - p1)
            o_width=int(np.floor(o_width))
            o_height = np.linalg.norm(p3 - p1)
            o_height=int(np.floor(o_height))

            design_h_plate = 53
            design_w_plate = 88
            float_range_x = 8
            float_range_y = 8
            resize_time = 6

            ori_cor = np.float32([p1, p2, p3, p4])
            dst_cor=np.float32([[0, 0],[o_width, 0],[0, o_height],[o_width, o_height]])    

            map_trans = cv2.getPerspectiveTransform(ori_cor, dst_cor)
            plate = cv2.warpPerspective(perspective_img, map_trans,(o_width, o_height))
            h_plate = plate.shape[0]
            w_plate = plate.shape[1]
            perspective_dst_path = 'output/perspective_dst.png'
            if flag_write:
                cv2.imwrite(perspective_dst_path, plate)         
            plate = cv2.resize(plate, (design_w_plate * resize_time, design_h_plate *resize_time))
            if flag_write:
                cv2.imwrite(perspective_dst_path, plate)         
                print('perspective dst:\n{}'.format(perspective_dst_path))
            
            # 牌照切割
            plate_section = Plate()
            plate_dict = {}
            output_dir = 'output/plate'

            # BRAND
            # BRAND_x_left = int(14.5 / design_w_plate * w_plate + 0.5 - float_range_x)
            # if BRAND_x_left < 0:
            #     BRAND_x_left = 0
            # BRAND_x_right = int(50.5 / design_w_plate * w_plate + 0.5 + float_range_x)
            # if BRAND_x_right > design_w_plate * resize_time:
            #     BRAND_x_right = design_w_plate * resize_time
            # BRAND_y_up = int(1 / design_h_plate * h_plate + 0.5 - float_range_y)
            # if BRAND_y_up < 0:
            #     BRAND_y_up = 0
            # BRAND_y_down = int(5 / design_h_plate * h_plate + 0.5 + float_range_y)
            # if BRAND_y_down > design_h_plate *resize_time:
            #     BRAND_y_down = design_h_plate *resize_time
            BRAND_x_left = 75
            BRAND_x_right = 295
            BRAND_y_up = 5
            BRAND_y_down = 40
            plate_section.BRAND = plate[BRAND_y_up: BRAND_y_down, BRAND_x_left: BRAND_x_right]
            if flag_write:
                cv2.imwrite(os.path.join(output_dir, 'brand.png'), plate_section.BRAND)

            # COUNTRY
            # COUNTRY_x_left = int(66.5 / design_w_plate * w_plate + 0.5 - float_range_x)
            # COUNTRY_x_right = int(88 / design_w_plate * w_plate + 0.5 + float_range_x)
            # COUNTRY_y_up = int(1 / design_h_plate * h_plate + 0.5 - float_range_y)
            # COUNTRY_y_down = int(5 / design_h_plate * h_plate + 0.5 + float_range_y)
            COUNTRY_x_left = 380
            COUNTRY_x_right = 615
            COUNTRY_y_up = 1
            COUNTRY_y_down = 35
            plate_section.COUNTRY = plate[COUNTRY_y_up: COUNTRY_y_down, COUNTRY_x_left: COUNTRY_x_right]
            if flag_write:
                cv2.imwrite(os.path.join(output_dir, 'country.png'), plate_section.COUNTRY)

            # TYPE
            # TYPE_x_left = int(20.5 / design_w_plate * w_plate + 0.5 - float_range_x)
            # TYPE_x_right = int(36.4 / design_w_plate * w_plate + 0.5 + float_range_x)
            # TYPE_y_up = int(6 / design_h_plate * h_plate + 0.5 - float_range_y)
            # TYPE_y_down = int(10 / design_h_plate * h_plate + 0.5 + float_range_y)
            TYPE_x_left = 110
            TYPE_x_right = 230
            TYPE_y_up = 30
            TYPE_y_down = 70
            plate_section.TYPE = plate[TYPE_y_up: TYPE_y_down, TYPE_x_left: TYPE_x_right]
            if flag_write:
                cv2.imwrite(os.path.join(output_dir, 'type.png'), plate_section.TYPE)

            # NUMBER
            # NUMBER_x_left = int(20.5 / design_w_plate * w_plate + 0.5 - float_range_x)
            # NUMBER_x_right = int(36.4 / design_w_plate * w_plate + 0.5 + float_range_x)
            # NUMBER_y_up = int(6 / design_h_plate * h_plate + 0.5 - float_range_y)
            # NUMBER_y_down = int(10 / design_h_plate * h_plate + 0.5 + float_range_y)
            NUMBER_x_left = 105
            NUMBER_x_right = 380
            NUMBER_y_up = 60
            NUMBER_y_down = 100
            plate_section.NUMBER = plate[NUMBER_y_up: NUMBER_y_down, NUMBER_x_left: NUMBER_x_right]
            if flag_write:
                cv2.imwrite(os.path.join(output_dir, 'number.png'), plate_section.NUMBER)

            # ENGINE MODEL
            ENGINE_x_left = 100
            ENGINE_x_right = 170
            ENGINE_y_up = 90
            ENGINE_y_down = 130
            plate_section.ENGINE = plate[ENGINE_y_up: ENGINE_y_down, ENGINE_x_left: ENGINE_x_right]
            if flag_write:
                cv2.imwrite(os.path.join(output_dir, 'engine.png'), plate_section.ENGINE)

            # POWER
            POWER_x_left = 160
            POWER_x_right = 250
            POWER_y_up = 120
            POWER_y_down = 160
            plate_section.POWER = plate[POWER_y_up: POWER_y_down, POWER_x_left: POWER_x_right]
            if flag_write:
                cv2.imwrite(os.path.join(output_dir, 'power.png'), plate_section.POWER)

            # GVM
            GVM_x_left = 150
            GVM_x_right = 280
            GVM_y_up = 150
            GVM_y_down = 190
            plate_section.GVM = plate[GVM_y_up: GVM_y_down, GVM_x_left: GVM_x_right]
            if flag_write:
                cv2.imwrite(os.path.join(output_dir, 'gvm.png'), plate_section.GVM)

            # ENGINE SIZE
            SIZE_x_left = 390
            SIZE_x_right = 570
            SIZE_y_up = 90
            SIZE_y_down = 130
            plate_section.SIZE = plate[SIZE_y_up: SIZE_y_down, SIZE_x_left: SIZE_x_right]
            if flag_write:
                cv2.imwrite(os.path.join(output_dir, 'size.png'), plate_section.SIZE)

            # DATEy
            DATEy_x_left = 325
            DATEy_x_right = 405
            DATEy_y_up = 120
            DATEy_y_down = 160
            plate_section.DATEy = plate[DATEy_y_up: DATEy_y_down, DATEy_x_left: DATEy_x_right]
            if flag_write:
                cv2.imwrite(os.path.join(output_dir, 'datey.png'), plate_section.DATEy)

            # DATEm
            DATEm_x_left = 430
            DATEm_x_right = 520
            DATEm_y_up = 120
            DATEm_y_down = 160
            plate_section.DATEm = plate[DATEm_y_up: DATEm_y_down, DATEm_x_left: DATEm_x_right]
            if flag_write:
                cv2.imwrite(os.path.join(output_dir, 'datem.png'), plate_section.DATEm)

            # OCCUPANTS
            OCCUPANTS_x_left = 370
            OCCUPANTS_x_right = 440
            OCCUPANTS_y_up = 150
            OCCUPANTS_y_down = 190
            plate_section.OCCUPANTS = plate[OCCUPANTS_y_up: OCCUPANTS_y_down, OCCUPANTS_x_left: OCCUPANTS_x_right]
            if flag_write:
                cv2.imwrite(os.path.join(output_dir, 'occupants.png'), plate_section.OCCUPANTS)

            # MANUFACTURER
            MANUFACTURER_x_left = 110
            MANUFACTURER_x_right = 460
            MANUFACTURER_y_up = 210
            MANUFACTURER_y_down = 250
            plate_section.MANUFACTURER = plate[MANUFACTURER_y_up: MANUFACTURER_y_down, MANUFACTURER_x_left: MANUFACTURER_x_right]
            if flag_write:
                cv2.imwrite(os.path.join(output_dir, 'manufacturer.png'), plate_section.MANUFACTURER)

            # MANUFACTURERn1
            MANUFACTURERn1_x_left = 100
            MANUFACTURERn1_x_right = 360
            MANUFACTURERn1_y_up = 240
            MANUFACTURERn1_y_down = 280
            plate_section.MANUFACTURERn1 = plate[MANUFACTURERn1_y_up: MANUFACTURERn1_y_down, MANUFACTURERn1_x_left: MANUFACTURERn1_x_right]
            if flag_write:
                cv2.imwrite(os.path.join(output_dir, 'manufacturer_n1.png'), plate_section.MANUFACTURERn1)

            # MANUFACTURERn2
            MANUFACTURERn2_x_left = 100
            MANUFACTURERn2_x_right = 320
            MANUFACTURERn2_y_up = 270
            MANUFACTURERn2_y_down = 310
            plate_section.MANUFACTURERn2 = plate[MANUFACTURERn2_y_up: MANUFACTURERn2_y_down, MANUFACTURERn2_x_left: MANUFACTURERn2_x_right]
            if flag_write:
                cv2.imwrite(os.path.join(output_dir, 'manufacturer_n2.png'), plate_section.MANUFACTURERn2)
            
            # 文字识别模块
            for each_plate_section in plate_section.__dict__.keys():
                ret, plate = cv2.threshold(getattr(plate_section, each_plate_section), 128, 255, cv2.THRESH_BINARY)
                plate = cv2.bitwise_not(plate)

                plate = cv2pil(plate)
                plate.save('output/converted.png')

                # 识别英文
                if each_plate_section in ['TYPE','ENGINE','POWER','GVM','SIZE','DATEy','DATEm','OCCUPANTS','MANUFACTURERn2']:
                    res = tool.image_to_string(
                        plate,
                        lang="eng",
                        builder=pyocr.builders.WordBoxBuilder(tesseract_layout=6))

                # 识别中文
                if each_plate_section in ['BRAND','COUNTRY','MANUFACTURER','MANUFACTURERn1','NUMBER']:
                    res = tool.image_to_string(
                        plate,
                        lang="chi_sim",
                        builder=pyocr.builders.WordBoxBuilder(tesseract_layout=6))

                if not len(res) == 0:
                    plate_dict[each_plate_section] = res[0].content
            
            # tuning
            if plate_dict['BRAND'] in '丰田(TOYOTA)':
                plate_dict['BRAND'] = '丰田(TOYOTA)'
                
            if plate_dict['COUNTRY'] in '中华人民共和':
                plate_dict['COUNTRY'] = '中华人民共和国'
            
            if plate_dict['MANUFACTURER'] in '天津一汽':
                plate_dict['MANUFACTURER'] = '天津一汽丰田汽车有限公司'
            
            plate_dict['DATEm'] = re.sub('[a-zA-Z]', '', plate_dict['DATEm'])
            plate_dict['NUMBER'] = re.sub('[a-z]', '', plate_dict['NUMBER'])

            if plate_dict['POWER'].endswith('k'):
                plate_dict['POWER'] = re.sub('k', 'kW', plate_dict['POWER'])
            
            if plate_dict['GVM'].endswith('k'):
                plate_dict['GVM'] = re.sub('k', 'kg', plate_dict['GVM'])

            if plate_dict['SIZE'].endswith('m'):
                plate_dict['SIZE'] = re.sub('m', 'ml', plate_dict['SIZE'])

            if plate_dict['DATEy'].endswith('¢'):
                plate_dict['DATEy'] = re.sub('¢', '9', plate_dict['DATEy'])

            print('----------\nrecognition compleate, result:\n{}'.format(plate_dict))
 
            # 得到识别成功的flag，便退出循环。
            # if flag_ok:
            #     break
            if not args:
                return plate_dict

@app.after_request
def cors(environ):
    environ.headers['Access-Control-Allow-Origin']='*'
    environ.headers['Access-Control-Allow-Method']='*'
    environ.headers['Access-Control-Allow-Headers']='x-requested-with,content-type,text.html'
    return environ

if __name__ == '__main__':
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    tool = tools[0]

    host_name = socket.gethostname()
    ip_add = socket.gethostbyname(host_name)

    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--sample_path')
    parser.add_argument('-F', '--offline', action='store_true', default=False)
    parser.add_argument('-H', '--host', default=ip_add, type=str)
    parser.add_argument('-P', '--port', default=5000, type=int)
    parser.add_argument('-D', '--debug_flask', action='store_true', default=False)
    parser.add_argument('-W', '--write_image', action='store_true', default=False)
    args = parser.parse_args()

    if args.offline:
        main(args, tool)

    else:
        app.config['tool'] = tool
        app.config['write_image'] = args.write_image
        app.run(host=args.host, port=args.port, debug=args.debug_flask)
