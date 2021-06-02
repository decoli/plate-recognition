import re


def adjust_plate(plate_dict):
    if plate_dict['BRAND'] in '丰田(TOYOTA)':
        plate_dict['BRAND'] = '丰田(TOYOTA)'
        
    if plate_dict['COUNTRY'] in '中华人民共和':
        plate_dict['COUNTRY'] = '中华人民共和国'
    
    if plate_dict['MANUFACTURER'] in '天津一汽':
        plate_dict['MANUFACTURER'] = '天津一汽丰田汽车有限公司'
    
    plate_dict['NUMBER'] = re.sub('[a-z]', '', plate_dict['NUMBER'])

    plate_dict['POWER'] = re.sub(r'\D', '', plate_dict['POWER'])
    plate_dict['DATEy'] = re.sub(r'\D', '', plate_dict['DATEy'])
    plate_dict['DATEm'] = re.sub(r'\D', '', plate_dict['DATEm'])
    plate_dict['SIZE'] = re.sub(r'\D', '', plate_dict['SIZE'])
    plate_dict['GVM'] = re.sub(r'\D', '', plate_dict['GVM'])
    
    return plate_dict