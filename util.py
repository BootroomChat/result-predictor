import http.client
import json
import os
import random
import uuid

import mtranslate
import requests

from config import *

KEYS_MAPPING = {'url': 'url', 'name': 'name', 'club': 'club', 'nation': 'nation', 'league': 'league',
                'skills': 'skills', 'weak foot': 'weakFoot', 'foot': 'foot', 'height': 'height', 'weight': 'weight',
                'revision': 'revision', 'def. wr': 'defenceAtt', 'att. wr': 'attackAtt', 'added on': 'addedDate',
                'origin': 'origin', 'r.face': 'hasRealFace', 'age': 'age', 'rating': 'rating',
                'displayName': 'displayName', 'intl. rep': 'intlRep', 'pictureImg': 'playerImg',
                'position': 'position', 'countryImg': 'nationImg', 'clubImg': 'clubImg', 'playerImg': 'playerImg',
                'id': 'id', 'cardImg': 'cardImg', 'PAC': 'PAC', 'SHO': 'SHO', 'PAS': 'PAS', 'DRI': 'DRI', 'DEF': 'DEF',
                'PHY': 'PHY', 'xboxMinPrice': 'xboxMinPrice', 'xboxMaxPrice': 'xboxMaxPrice',
                'psMinPrice': 'psMinPrice', 'psMaxPrice': 'psMaxPrice', 'pcMinPrice': 'pcMinPrice',
                'pcMaxPrice': 'pcMaxPrice', 'pos': 'position', 'ski': 'skills'}

DEFAULT_IMAGE_URL = 'https://i.imgur.com/slYAXU0.jpg'


def load_json(file_name, default=dict):
    if os.path.exists(file_name):
        with open(file_name) as json_data:
            d = json.load(json_data)
            return d
    else:
        return default()


def write_json(file_name, json_data):
    print('writing:' + file_name)
    with open(file_name, 'w') as outfile:
        json.dump(json_data, outfile, ensure_ascii=False)
        print('writing done:' + file_name)
        return True


def ordered(obj):
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items() if k is not None and v is not None)
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj if x is not None)
    if obj is None:
        return ""
    else:
        return obj


def data_changed(local_item, item):
    return ordered(local_item) != ordered(item)


def upload_image(im, origin_url, name, default_url):
    try:
        print(origin_url)
        origin_url = origin_url.replace(
            'https://imgsa.baidu.com/', 'http://imgsa.baidu.com/')
        available_proxies = ["http://222.221.11.119:3128",
                             'http://115.238.59.86:60635', 'http://119.57.105.25:53281']
        proxies = {"http": random.choice(available_proxies)}
        try:
            img_res = requests.get(origin_url, proxies=proxies, timeout=30)
        except Exception as e:
            print(e)
            try:
                img_res = requests.get(origin_url, proxies=proxies, timeout=30)
            except Exception as e:
                print(e)
                img_res = requests.get(origin_url, timeout=15)
        uploaded_image = requests.post('https://sm.ms/api/upload', {},
                                       files={'smfile': img_res.content}, timeout=20, headers=headers).json()
        return uploaded_image['data']['url']
    except Exception as e:
        try:
            uploaded_image = im.upload_image(url=origin_url, title=name)
            return uploaded_image.link
        except Exception as e:
            print(e)
            return default_url


host = 'api.cognitive.microsofttranslator.com'
path = '/translate?api-version=3.0'

params = "&to=zh-Hans"


def translate(text):
    request_body = [{'Text': text}]
    content = json.dumps(request_body, ensure_ascii=False).encode('utf-8')
    headers = {
        'Ocp-Apim-Subscription-Key': translate_key1,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    conn = http.client.HTTPSConnection(host)
    result = text
    try:
        result = mtranslate.translate(text, "zh-cn", "auto")
    except Exception as e:
        print(e)
        try:
            conn.request("POST", path + params, content, headers)
            result = json.loads(conn.getresponse().read())[
                0]['translations'][0]['text']
        except Exception as ee:
            print(ee)
    return result
