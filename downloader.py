#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import sys
import glob
import os
import time
import datetime
import re
import threading
import urllib.parse
import urllib.request

class DownloaderThread(threading.Thread):
    def __init__(self, threadId, urlList=[]):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.urlList = urlList

    def run(self):
        print(datetime.datetime.now(), self.threadId, "start to download")
        for url in self.urlList:
            filename = url[url.rfind("/")+1:]
            if os.path.exists(filename):
                continue
            print(datetime.datetime.now(), self.threadId, "wget %s" %(url))
            os.system("wget %s" %(url))   	    	
            time.sleep(1)
        print(datetime.datetime.now(), self.threadId, "completed")

class Downloader:
    def __init__(self, m3u8, nThread=10):
        self.baseUrl = m3u8[:m3u8.rfind("/")+1]
        self.m3u8 = m3u8
        self.nThread=nThread
        self.lines = self.crawl_binary(m3u8).split("\n")

    def crawl_binary(self, url):
        print(datetime.datetime.now(), "download list")
        data =  urllib.request.urlopen(url).read()
        with open("index.m3u8","wb") as f:
            f.write(data)
        return data.decode("utf-8")

    def download(self):
        threads = []
        urlLists = []
        for i in range(self.nThread):
            urlLists.append([])

        i = 0
        for line in self.lines:
            #if i>10:
            #    break
            if line.endswith("ts"):
                threadId = i % self.nThread
                targetUrl = urllib.parse.urljoin(self.baseUrl, line)

                print(datetime.datetime.now(), threadId, "added url: ", targetUrl)
                urlLists[threadId].append(targetUrl)
                i += 1

        for i in range(self.nThread):
            thread = DownloaderThread(i, urlLists[i])
            threads.append(thread)
            threads[-1].start()

        for i in range(self.nThread):
            threads[i].join()

        print(datetime.datetime.now(), "merge ts files")
        targetFilename = os.path.abspath(".") + ".ts"
        for line in self.lines:
            if line.endswith("ts") and os.path.exists(os.path.basename(line)):
                print(datetime.datetime.now(), "cat %s >> %s" % (os.path.basename(line), targetFilename))
                os.system("cat %s >> %s" % (os.path.basename(line), targetFilename))


if __name__ == "__main__":
    import locale
    import os
    from dialog import Dialog
    # pip3 install pythondialog
    if len(sys.argv) == 2:
        m3u8 = sys.argv[1]
        Downloader(m3u8, nThread=25).download()
    else:
        locale.setlocale(locale.LC_ALL, '') 
        d = Dialog(dialog="dialog") 
        d.set_background_title("Downloader")

        def input_dialog(title):
            res = d.inputbox(title, 7, 40)
            if res[0] == 'ok':
                return res[1]
            return None

        m3u8 = input_dialog("Input m3u8 url:")
        dirName =  input_dialog("Input dirname:")

        script = '''#!/bin/bash

V_NAME=%s
M3U8_URL=%s

mkdir $V_NAME
cp downloader.py $V_NAME
cd $V_NAME
python3.9 downloader.py ${M3U8_URL}
cd ..
ffmpeg -y -threads 0 -i ${V_NAME}.ts -vf scale=720:406 ${V_NAME}.mp4
rm -Rf ${V_NAME}
rm  ${V_NAME}.ts
''' % (dirName, m3u8)
        d.msgbox(script,20,60)

        with open("run.sh", "w") as f:
            f.write(script)
        os.system("chmod 777 *.sh")
        os.system("clear")
        os.system("./run.sh")
