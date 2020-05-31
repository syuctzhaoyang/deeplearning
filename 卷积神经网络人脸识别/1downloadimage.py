# _*_ coding: utf-8 _*_
import requests
from urllib import parse
import os

#设置访问头

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'}

#设置访问地址，word={idolname}&step_word={idolname}&pn={pages}为百度查找关键字，以及后续页数
#注：百度图片30张为一页，pn=0为前30张，pn=1为第31张至第60张

img_url = 'http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&is=&fp=result&word={idolname}&step_word={idolname}&pn={pages}'

# keyword为所查找图片的关键字, dest_dir下载到本地存储的路径, batch下载的批次，一个批次为30张

def getIdolPicture(keyword, dest_dir, batch):

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    for i in range(batch):
        try:
            #parse.quote(keyword)功能为将汉字翻译成UTF-8编码
            #img_url.format将keyword，pages参数融入字符串中
            #调用requests库中的get方法，爬取百度图片,返回为一个requests.models.Response对象
            res = requests.get(img_url.format(idolname=parse.quote(keyword), pages=i * 30))
            #设置对象编码格式为utf-8，
            res.encoding = 'utf-8',
            #将res解析成json对象，取['data']，返回为一个列表
            #遍历列表，取出每个元素中thumbURL项的值，即图片的实际存储地址
            #注：百度网页中放置图片的还有ObjURL
            #此处可以使用Xpath,正则，或是beautiful soup取得图片地址
            for ele in res.json()['data']:
                url = ele.get('thumbURL')
                #如果url中的值不为空，则使用图片原有的名称为其名
                if url:
                    #按二进制文件存储
                   with open(dest_dir + url.split('/')[-1], 'wb') as f:
                       # 调用requests库中的get方法，爬取百度图片，返回为一个requests.models.Response对象
                       #Response对象中content为byte类型数据
                       #注：res2.content.decode('utf-8')=res2.text
                        res2 = requests.get(url, headers = headers)
                       #将下载后的图片存储在dest_dir目录下
                        f.write(res2.content)
        #此处防止网络问题导致图片下载不了的情况造成的死机情况，下载不了就进入下个循环，相对于continue
        except:
            print("该条记录无法下载")
#下载范冰冰照片，300张
getIdolPicture('范冰冰', 'idol1/',10)
getIdolPicture('迪丽热巴', 'idol2/',10)
getIdolPicture('林志玲', 'idol3/',10)



