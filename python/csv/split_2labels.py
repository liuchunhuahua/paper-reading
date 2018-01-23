#!/usr/bin/env python
#encoding=utf-8
#date: 20180123 give true: pred: and othes ,split by confuse matrix
# step1: classify the input file to 0 categories, and write to each .csv
# step2: combine 9 .csv to nine_categories.xls
# step3: rm 9.csv
import csv
import xlwt
import os

def load_data(file_name):

  #f00 = open("00.csv","w")
  f00 = csv.writer(open("00.csv","w+"),lineterminator='\n')
  f01 = csv.writer(open("01.csv", "w+"),lineterminator='\n')
  f02 = csv.writer(open("02.csv", "w+"),lineterminator='\n')

  f10 = csv.writer(open("10.csv", "w+"),lineterminator='\n')
  f11 = csv.writer(open("11.csv", "w+"),lineterminator='\n')
  f12 = csv.writer(open("12.csv", "w+"),lineterminator='\n')

  f20 = csv.writer(open("20.csv", "w+"),lineterminator='\n')
  f21 = csv.writer(open("21.csv", "w+"),lineterminator='\n')
  f22 = csv.writer(open("22.csv", "w+"),lineterminator='\n')
  count={'00':0,'01':0,'02':0,'10':0,'11':0,'12':0,'20':0,'21':0,'22':0}
  with open (file_name) as f:
    for l in f:

      line= l.strip("\n").split("	")
      #l_out = str(line[0]) +"\t"+str(line[1])+"\t"+str(line[2])+"\t"+str(line[3])+"\t"+str(line[4])+ "\n"
      #l_out=str(line[0]) +"\t"+str(line[1])+"\t"+str(line[5])+ "\n"
      l_out=(line[0],line[1],line[2])
      if (line[0]=="true:0" and line[1]=="pred:0"):
        #f00.write(l_out)
        f00.writerow(l_out)
        count['00'] +=1
      elif (line[0]=="true:0" and line[1]=="pred:1"):
        f01.writerow(l_out)
        count['01'] += 1
      elif (line[0]=="true:0" and line[1]=="pred:2"):
        f02.writerow(l_out)
        count['02'] += 1
      elif (line[0]=="true:1" and line[1]=="pred:0"):
        f10.writerow(l_out)
        count['10'] += 1
      elif (line[0]=="true:1" and line[1]=="pred:1"):
        f11.writerow(l_out)
        count['11'] += 1
      elif (line[0]=="true:1" and line[1]=="pred:2"):
        f12.writerow(l_out)
        count['12'] += 1

      elif (line[0]=="true:2" and line[1]=="pred:0"):
        f20.writerow(l_out)
        count['20'] += 1
      elif (line[0]=="true:2" and line[1]=="pred:1"):
        f21.writerow(l_out)
        count['21'] += 1
      elif (line[0]=="true:2" and line[1]=="pred:2"):
        f22.writerow(l_out)
        count['22'] += 1
    for key in sorted(count.keys()):
      print (key,count[key])




def combine_csv(xlsfile=None):
  print ("combine csv files !!!")
  if xlsfile==None:
    xlsfile = './result.xls' #目标execl文件名

  workbook = xlwt.Workbook()  #初始化workbook对象

  filelist = sorted(os.listdir('./')) #创建当前目录下文件列表
  for file in filelist:

    if file.endswith('.csv'): #匹配以csv结尾的文件
        sheetname = file.replace('.csv', '') #匹配出表名

        write_execl(file, xlsfile, workbook, sheetname) #使用函数将csv文件内容导入到execl
  workbook.save(xlsfile)

def write_execl(csvfile, xlsfile, workbook, sheetname):

  sheet = workbook.add_sheet(sheetname)  # 创建表名，添加一个workbook的对象

  reader = csv.reader(open(csvfile, 'r'))  # 读取csv文件内容，写入表
  i = 0
  for content in reader:
    for j in range(len(content)):
      sheet.write(i, j, content[j])

      j += 1
    i += 1
  print (csvfile,"row:", i, "col:", j)

def remove_files():
  print ("remove the ubnecessary files!!")
  os.remove("00.csv")
  os.remove("01.csv")
  os.remove("02.csv")
  os.remove("10.csv")
  os.remove("11.csv")
  os.remove("12.csv")
  os.remove("20.csv")
  os.remove("21.csv")
  os.remove("22.csv")


def main():
  load_data("label_test.txt")
  combine_csv(xlsfile="nine_categories.xls")
  remove_files()
  print ("done!!!")

if __name__ == '__main__':
  main()