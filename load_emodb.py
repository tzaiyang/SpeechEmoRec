#encoding:utf-8
import urllib
import zipfile
import os
import shutil
import sys
from six.moves import urllib
import tarfile
import path

def load_data(server_file,source_url, obj_directory):
    if not os.path.exists(server_file):
        tarpath = maybe_download(server_file,source_url)

    if not os.path.exists(obj_directory):
        untar(server_file, server_file.split('.')[0])
        build_class_directories(server_file,obj_directory)

def maybe_download(filename, source_url):
    datadir_root = ''.join(filename.split('/')[:-1])
    filepath = ''
    if not os.path.exists(datadir_root): 
        os.makedirs(datadir_root)
    if not os.path.exists(filename):
        print("Downloading %s from %s, Please "
              "wait..."%(filename,source_url))
        filepath, _ = urllib.request.urlretrieve(source_url,
                                                 filename, reporthook)
        statinfo = os.stat(filename)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')

    return filepath

#reporthook from stackoverflow #13881092
def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))

def build_class_directories(filename,dir):
    os.rename('%s/wav'%filename.split('.')[0],dir)
    shutil.rmtree(filename.split('.')[0])

def classfier_dataset_todir(Dataset_path):
    wave_filenames = os.listdir(Dataset_path)
    for filename in wave_filenames:
        if not os.path.exists('%s/%s/%s'%(Dataset_path,filename[:2],filename[5:6])):
            os.makedirs('%s/%s/%s'%(Dataset_path,filename[:2],filename[5:6]))
        #os.rename('%s/%s'%(Dataset_path,filename),'%s/%s/%s/%s'%(Dataset_path,filename[:2],filename[2:5],filename[5:12]))

def untar(fname, extract_dir):
    if fname.endswith("tar.gz") or fname.endswith("tgz"):
        tar = tarfile.open(fname)
        tar.extractall(extract_dir)
        tar.close()
        print("File Extracted")

    elif fname.endswith("zip"):
        zip = zipfile.ZipFile(fname, 'r')
        zip.extractall(extract_dir)
        # for f in zip.namelist():
        #     print(f)
        zip.close()
        print("File Extracted")

    else:
        print("Not a tar.gz/tgz/zip file: '%s '" % sys.argv[0])

if __name__ == "__main__":
    url = 'http://www.emodb.bilderbar.info/download/download.zip'
    server_fname = url.split('/')[-1]
    load_data(server_file=path.DataDir.DataRoot+server_fname,source_url=url,obj_directory=path.DataDir.wav)
