# _*_ coding: utf-8 _*_
import requests
from urllib import parse
import os

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'}
img_url = 'http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&is=&fp=result&word={idolname}&step_word={idolname}&pn={pages}'

def getIdolPicture(keyword, dest_dir, batch):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    for i in range(batch):
        try:
            url = img_url.format(idolname=parse.quote(keyword), pages=i * 30)
            res = requests.get(img_url.format(idolname=parse.quote(keyword), pages=i * 30))

            res.encoding = 'utf-8',
            for ele in res.json()['data']:
                url = ele.get('thumbURL')
                if url:
                   with open(dest_dir + url.split('/')[-1], 'wb') as f:
                        res2 = requests.get(url, headers = headers)
                        f.write(res2.content)
        except:
            print("该条记录无法下载")

getIdolPicture('范冰冰', 'idol1/',10)
getIdolPicture('迪丽热巴', 'idol2/',10)
getIdolPicture('林志玲', 'idol3/',10)



