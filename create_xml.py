import os
from xml.dom.minidom import *
import pymysql
from PIL import Image
import urllib.request
import urllib.parse
from tqdm import tqdm
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from DBUtils.PooledDB import PooledDB
# from dbutils.pooled_db import PooledDB
import zipfile

conn = pymysql.connect(host='', port=, user='', password='',db='l')
cur = conn.cursor()
app = FastAPI()

mysqlInfo = {
    "host": '',
    "user": '',
    "passwd": '',
    "db": '',
    "port": ,
    "connect_timeout": 
}


class ConnMysql(object):
    __pool = None

    def __init__(self):
        # 构造函数，创建数据库连接、游标
        self.coon = ConnMysql.getmysqlconn()
        self.cur = self.coon.cursor(cursor=pymysql.cursors.DictCursor)

    # 数据库连接池连接
    @staticmethod
    def getmysqlconn():
        global __pool
        if ConnMysql.__pool is None:
            __pool = PooledDB(
                creator=pymysql,
                mincached=2,
                maxcached=5,
                maxconnections=6,
                maxshared=3,
                blocking=True,
                maxusage=None,
                setsession=[],
                ping=0,
                host=mysqlInfo['host'],
                user=mysqlInfo['user'],
                passwd=mysqlInfo['passwd'],
                db=mysqlInfo['db'],
                port=mysqlInfo['port'])
        return __pool.connection()

    # 插入、修改、删除一条
    def sql_change_msg(self, sql):
        change_sql = self.cur.execute(sql)
        self.coon.commit()
        return change_sql

    # 查询一条
    def sql_select_one(self, sql):
        self.cur.execute(sql)
        select_res = self.cur.fetchone()
        return select_res

    # 查询多条
    def sql_select_many(self, sql, count=None):
        print(self.coon)
        self.cur.execute(sql)
        if count is None:
            select_res = self.cur.fetchall()
        else:
            select_res = self.cur.fetchmany(count)
        return select_res

    # 释放资源
    def release(self):
        self.cur.close()
        self.coon.close()


mysqlobj = ConnMysql()


class Model(BaseModel):
    datasetId: str = None
    imageState: str = None
    saveRootLocal: str = None
    groupId: list = None
    no_package_label: list = None


def create_zip(datasetId: int, image_state: str, group_id: int, save_local_root_path: str,no_package_label: list) -> None:
    if not os.path.exists(save_local_root_path):
        os.makedirs(save_local_root_path)

    img_save_path_file = os.path.join(save_local_root_path, 'src')
    if not os.path.exists(img_save_path_file):
        os.makedirs(img_save_path_file)

    # for i in range(len(group_id) - 1):

    if group_id is None:

        sql = 
        cur.execute(sql)
        # 提交事务
        conn.commit()
        imgInfos = [[i[0], i[1], i[2]] for i in cur.fetchall()]  # imgId, path, image_name
    else:
        # mysql 内连接查询
        sql2 = 

        cur.execute(sql2)
        conn.commit()
        imgInfos = [[i[0], i[1], i[2]] for i in cur.fetchall()]  # imgId, path, image_name

    for order, img in enumerate(tqdm(imgInfos, desc="====生成xml文件中", ncols=150, nrows=10)):
        doc = Document()
        # 创建一个根节点
        root = doc.createElement('annotation')
        # 根节点加入到tree
        doc.appendChild(root)

        folder = doc.createElement('folder')
        folder.appendChild(doc.createTextNode('src'))
        root.appendChild(folder)

        filename = doc.createElement('filename')
        filename.appendChild(doc.createTextNode('{}'.format(img[2])))  # 图片名称
        root.appendChild(filename)

        img_id = img[0]
        # img_name = str(img_id) + "_" + img[2]
        img_name = img[2]
        img_path = img[1]

        # img_save_path = './jxsf_bowl1/img/{}'.format(img_name)
        img_save_path = os.path.join(img_save_path_file, img_name)

        path = doc.createElement('path')
        # img_save_path_bak = "D:\src\jxsf_bowl1\src\{}".format(img_name)
        img_save_path_bak = "{}".format(img_save_path)
        path.appendChild(doc.createTextNode('{}'.format(img_save_path_bak)))  # 本地图片路经
        root.appendChild(path)
        source = doc.createElement('source')

        database = doc.createElement('database')
        # database.appendChild(doc.createTextNode('{}'.format(datasetId)))  # 数据集
        database.appendChild(doc.createTextNode("Unknown"))  # 数据集
        source.appendChild(database)
        root.appendChild(source)

        # 2
        img = img[1]
        if "http" in img:
            print("===is_url_img===", img)
            img = Image.open(urllib.request.urlopen(img))  # 图片网页地址
            print(img.size, img.mode)  # (924, 718) RGB  -->whc
        else:
            try:
                img = Image.open(img).convert('RGB')
            except Exception as err:
                print("============err==============", err)
                continue

        size = doc.createElement('size')

        width = doc.createElement('width')
        width.appendChild(doc.createTextNode('{}'.format(img.size[0])))  # 图片的宽

        height = doc.createElement('height')
        height.appendChild(doc.createTextNode('{}'.format(img.size[1])))  # 图片的高

        depth = doc.createElement('depth')
        depth.appendChild(doc.createTextNode(str(3)))  # 图片的深

        size.appendChild(width)
        size.appendChild(height)
        size.appendChild(depth)
        root.appendChild(size)

        segmented = doc.createElement('segmented')
        segmented.appendChild(doc.createTextNode('0'))
        root.appendChild(segmented)

        # url_path = doc.createElement('url_path')
        # url_path.appendChild(doc.createTextNode('{}'.format(img[1])))  # 网络图片路经
        # root.appendChild(url_path)

        # 3  循环进行添加
        sql1 = 
        cur.execute(sql1)
        conn.commit()
        annoInfos = [[i[0], i[1], i[2], i[3], i[4]] for i in cur.fetchall()]  # imgId, path, image_name
        # [[99, 320, 77, 341, 'JXSFJG-Chang-Fang-Xing-Die'], [259, 520, 378, 632, 'JXSFJG-Si-Fang-Wan'], [319, 594, 93, 366, 'JXSFJG-Si-Fang-Wan']]

        for order, obj in enumerate(annoInfos):
            # 只封装有效标签
            if obj[4] in no_package_label:
                print("==图片{}=目标为{}=obj[4]===".format(img_id, obj[4]))
                continue

            object = doc.createElement('object')
            bndbox = doc.createElement('bndbox')

            pose = doc.createElement('pose')
            pose.appendChild(doc.createTextNode('Unspecified'))

            order = doc.createElement('order')
            order.appendChild(doc.createTextNode('{}'.format(order)))

            truncated = doc.createElement('truncated')
            truncated.appendChild(doc.createTextNode('0'))

            difficult = doc.createElement('difficult')
            difficult.appendChild(doc.createTextNode('0'))

            name = doc.createElement('name')
            name.appendChild(doc.createTextNode("{}".format(obj[4])))
            # name.appendChild(doc.createTextNode("{}".format(label_dict[obj[4]])))

            xmin = doc.createElement('xmin')
            ymin = doc.createElement('ymin')
            xmax = doc.createElement('xmax')
            ymax = doc.createElement('ymax')

            xmin.appendChild(doc.createTextNode("{}".format(obj[0])))
            ymin.appendChild(doc.createTextNode("{}".format(obj[1])))
            xmax.appendChild(doc.createTextNode("{}".format(obj[2])))
            ymax.appendChild(doc.createTextNode("{}".format(obj[3])))

            object.appendChild(name)
            object.appendChild(pose)
            object.appendChild(truncated)
            object.appendChild(difficult)

            object.appendChild(bndbox)
            bndbox.appendChild(xmin)
            bndbox.appendChild(ymin)
            bndbox.appendChild(xmax)
            bndbox.appendChild(ymax)

            root.appendChild(object)

        label_path = os.path.join(save_local_root_path, 'label')
        if not os.path.exists(label_path):
            os.mkdir(label_path)

        try:
            # 下载图片存放
            urllib.request.urlretrieve(img_path, img_save_path)
            # 生成xml
            label_path_xml = os.path.join(label_path, img_name.split('.')[0] + ".xml")

            fp = open(label_path_xml, 'w', encoding='utf-8')
            try:
                doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding='utf-8')
            finally:
                fp.close()
        except Exception as err:
            print("err", err)
            continue

@app.post('/create_xml/saveLocal')
def create_xml(model: Model):
  
    print("==============create_xml saveLocal request is success===============")
    datasetId = model.datasetId
    image_state = model.imageState
    groupid = model.groupId
    save_local_root_path = model.saveRootLocal
    no_package_label = model.no_package_label
    print(datasetId, image_state, groupid, save_local_root_path, no_package_label)
    for group_id in groupid:
        create_zip(datasetId, image_state, group_id, save_local_root_path, no_package_label)
    return "ok"

if __name__ == '__main__':
    uvicorn.run(app='create_xml1:app', host='0.0.0.0', port=xx, debug=True)
