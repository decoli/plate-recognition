import os
import re

import cv2

from plate import Plate


def get_plate_section(plate, flag_write):
    output_dir = 'output/plate'
    plate_section = Plate()
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

    return plate_section