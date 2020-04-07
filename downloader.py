# -*- coding: utf-8 -*-
import sys
import glob
import os
import time
import datetime
import re
import threading
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
            if i>10:
                break
            if line.endswith("ts"):
                threadId = i % self.nThread
                print(datetime.datetime.now(), threadId, "added url: ", self.baseUrl+line)
                urlLists[threadId].append(self.baseUrl+line)
                i += 1

        for i in range(self.nThread):
            thread = DownloaderThread(i, urlLists[i])
            threads.append(thread)
            threads[-1].start()

        for i in range(self.nThread):
            threads[i].join()

        print(datetime.datetime.now(), "merge ts files")
        for line in self.lines:
            if line.endswith("ts") and os.path.exists(line):
                print(datetime.datetime.now(), "cat %s >> all.ts" % line)
                os.system("cat %s >> all.ts" % line)


if __name__ == "__main__":
    m3u8 = sys.argv[1]
    Downloader(m3u8, nThread=10).download()

