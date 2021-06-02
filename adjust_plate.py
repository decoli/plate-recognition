import re


def adjust_plate(plate_dict):
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

    return plate_dict