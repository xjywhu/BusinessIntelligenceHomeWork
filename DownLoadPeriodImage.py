import pandas as pd
import urllib.request
import os


def get_image_url(number):
  csv_data = pd.read_excel('catalog.xlsx')
  urls = reversed(csv_data["URL"])
  times = reversed(csv_data["TIMEFRAME"])
  url_dic = {}
  number1 = 0
  number2 = 0
  number3 = 0
  number4 = 0
  for (url, time) in zip(urls, times):
    url = url.replace("html", "detail", 1).replace("html", "jpg", 1)
    if time == "1551-1600" and number1 <= number:
      # 后文艺复兴时期
      url_dic[url] = "After Renaissance"
      number1 = number1+1
    elif time == "1400-1450" or time == "1451-1500" and number2 <= number:
      # 文艺复兴时期
      url_dic[url] = "Renaissance"
      number2 = number2+1
    elif time == "0701-0750" or time == "0751-0800" and number3 <= number:
      # 中世纪
      url_dic[url] = "Middle"
      number3 = number3+1
    elif time == "1751-1800" and number4 <= number:
      # 近现代
      url_dic[url] = "Modern"
      number4 = number4+1
  # print(number1, number2, number3, number4)
  return url_dic


def download_img(number):
  api_token = "fklasjfljasdlkfjlasjflasjfljhasdljflsdjflkjsadljfljsda"
  url_dic = get_image_url(number)
  print(type(url_dic))
  header = {"Authorization": "Bearer " + api_token}  # 设置http header
  for key in url_dic.keys():
    request = urllib.request.Request(key, headers=header)
    # try:
    img_name = key[key.rindex("/"):len(key)]
    print(key)
    response = urllib.request.urlopen(request)
    filename = "./image/" + url_dic[key]+img_name
    if not os.path.exists("./image/" + url_dic[key]):
      os.makedirs("./image/" + url_dic[key])
    if (response.getcode() == 200):
        with open(filename, "ab") as f:
            f.write(response.read())# 将内容写入图片
            print(img_name+"下载成功")

# download_img(20)


def download_Renaissance():
  api_token = "fklasjfljasdlkfjlasjflasjfljhasdljflsdjflkjsadljfljsda"
  header = {"Authorization": "Bearer " + api_token}  # 设置http header
  url = "https://img.ivsky.com/img/tupian/pre/201111/22/wenyifuxing_youhua-0"
  for i in range(1,32):
    if i <= 9:
      id = str(0)+str(i)
    else:
      id = str(i)
    img_url = url+id+".jpg"
    print(img_url)
    request = urllib.request.Request(img_url, headers=header)
    # try:
    img_name = id
    response = urllib.request.urlopen(request)
    filename = "./image/Renaissance/" + img_name +".jpg"
    if not os.path.exists("./image/Renaissance"):
      os.makedirs("./image/Renaissance")
    if (response.getcode() == 200):
      with open(filename, "ab") as f:
        f.write(response.read())  # 将内容写入图片
        print(img_name + "下载成功")
download_Renaissance()