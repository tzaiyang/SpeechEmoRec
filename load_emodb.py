#encoding:utf-8
import urllib
import zipfile
import os
import shutil
import sys
from six.moves import urllib
import tarfile

def load_data(DatasetDir,filename,source_url, obj_directory):
    dataset_file = os.path.join(DatasetDir, filename)
    if not os.path.exists(dataset_file):
        tarpath = maybe_download(filename,source_url,DatasetDir,obj_directory)

    if not os.path.exists(obj_directory):
        untar(filename, filename.split('.')[0])
        build_class_directories(filename,obj_directory)

def maybe_download(filename, source_url, work_directory,obj_directory):
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        print("Downloading Dataset from %s, Please "
              "wait..."%source_url)
        filepath, _ = urllib.request.urlretrieve(source_url,
                                                 filepath, reporthook)
        statinfo = os.stat(filepath)
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
    # dir_id = 0
    # class_dir = os.path.join(dir, str(dir_id))
    # if not os.path.exists(class_dir):
    #     os.mkdir(class_dir)
    # for i in range(1, 1361):
    #     fname = "image_" + ("%.4i" % i) + ".jpg"
    #     os.rename(os.path.join(dir, fname), os.path.join(class_dir, fname))
    #     if i % 80 == 0 and dir_id < 16:
    #         dir_id += 1
    #         class_dir = os.path.join(dir, str(dir_id))
    #         os.mkdir(class_dir)
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
    load_data(DatasetDir="./Dataset", filename=server_fname, source_url=url,obj_directory='Dataset/EMODB')