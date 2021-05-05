import os
import sys
import wavio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


ffmpegBinary = "\"%s\"" %  """C:\\\\Program Files\\\\ffmpeg\\\\bin\\\\ffmpeg.exe"""
soxBinary =  "\"%s\"" % """C:\\\\Program Files (x86)\\\\sox-14-4-2\\\\sox.exe"""

def to_wav(mp4Filename, wavFilename):
  if os.path.exists(wavFilename):
    return
  tmpWav = wavFilename+".wav"
  cmd = "%s -y -i %s %s" %(ffmpegBinary, mp4Filename, tmpWav)
  print(cmd)
  os.system(cmd)

  cmd = "%s %s -c 1 -r 8000 %s" % (soxBinary, tmpWav, wavFilename)
  print(cmd)
  #subprocess.call([cmd])
  os.system(cmd)

  cmd = "del %s" % tmpWav
  print(cmd)
  os.system(cmd)

def parse(wavFilename):
  w = wavio.read(wavFilename)
  output = []
  data = w.data
  wSize = 160
  threshold = 100
  for i in range(data.shape[0] // wSize):
    startPos, endPos = i*wSize, (i+1)* wSize

    chunk = data[startPos:endPos]
    stats = np.mean(chunk), np.max(np.abs(chunk))

    if (stats[1] <= threshold ):
      #print(startPos/w.rate, endPos/w.rate, stats)
      output.append([0, stats[1], stats[0]])
    else:
      output.append([1, stats[1], stats[0]])
  output = np.array(output)


  gap = int(0.1 * w.rate) // wSize
  X = []
  for i in range(output.shape[0]):
    if (output[i,0] == 0):
        X.append([output[i,0], i])

  X = np.array(X)

  clustering = DBSCAN(eps=10, min_samples=2).fit(X)
  clusters, currLabel = [], -1
  for i in range(len(clustering.labels_)):
    if currLabel != clustering.labels_[i]:
      clusters.append([X[i][1], X[i][1]])
      currLabel = clustering.labels_[i]
    else:
      clusters[-1][1] = X[i][1]
  


  for i in range(len(clusters)):
    a = int(float(clusters[i][0] * wSize) / w.rate)
    clusters[i] = ("%2d:%2d:%2d"%(a //3600, (a - a//3600 *3600 )//60, (a % 60))).replace(" ","0")

  print("\r\n".join(clusters))

  a = int(data.shape[0] / w.rate)
  print(("%2d:%2d:%2d"%(a //3600, (a - a//3600 *3600 )//60, (a % 60))).replace(" ","0"))
  
  

  
  plt.plot(output[:,2])
  plt.plot(output[:,0] * 1600)
  plt.show()



def main(argv):
  
  mp4Filename= argv[1]
  wavFilename = "w.wav"
  to_wav(mp4Filename, wavFilename)
  parse(wavFilename)

if __name__ == "__main__":
    main(sys.argv)
