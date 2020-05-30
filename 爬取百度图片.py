# -*- coding: utf-8 -*-
import os
import re
import time

import requests

num = 0
numPicture = 0
file = ''
List = []
word = ''


def Many_urls(Url, num):
    print('正在检测图片，请稍等.....')
    t = 0
    List = []

    while t < (int(num / 30) + 1):
        url = Url + str(t)
        t += 1
        try:
            urls = requests.get(url)
            urls.encoding = 'utf-8'
            text = urls.text
            pic_url = re.findall('"objURL":"(.*?)",', text, re.S)  # 先利用正则表达式找到图片url
            print(pic_url)
            if len(pic_url) == 0:
                break
            else:
                List.extend(pic_url)

        except:
            print('打开网页失败')
            continue

    return List


def pic_content(url):
    try:
        pic = requests.get(url)

    except:
        print('图片获取失败')
    return pic.content


def pic_download(content, string):
    fp = open(string, 'wb')
    fp.write(content)
    fp.close()


if __name__ == '__main__':  # 主函数入口

    word = input("请输入搜索关键词(可以是人名，地名等): ")

    # url = 'http://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&sf=1&fmq=&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&fm=result&pos=history&word=' + word + '&pn='
    url = 'http://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word=' + word + '&pn='
    numPicture = int(input('请输入想要下载的图片数量 '))
    urls = Many_urls(url, numPicture)

    print('检测完成！！！！！！！！！！！！！')
    time.sleep(1)

    file = input('请建立一个存储图片的文件夹，输入文件夹名称即可')

    if os.path.exists(file):
        print('该文件已存在，请重新输入')
        file = input('请建立一个存储图片的文件夹，)输入文件夹名称即可')
        os.mkdir(file)
    else:
        os.mkdir(file)

    num = 1
    print('找到关键词:' + word + '的图片，即将开始下载图片...')
    for url in urls[:numPicture]:
        try:
            string = file + r'\\' + word + '_' + str(num) + '.jpg'
            print('正在下载第' + str(num) + '张图片，图片地址:' + url)
            content = pic_content(url)
            pic_download(content, string)
            num += 1
        except:
            continue

    print('当前搜索结束，感谢使用')
    print('猜你喜欢')
