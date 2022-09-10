import gdown
import zipfile
import logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(funcName)s - %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
gdown.download(id="1UtXsUUuRt_17NMUMW5oAt5uENDNjGTiW",output="Final_pics.zip")
local_zip = 'Final_pics.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall()
zip_ref.close()
logging.info("extracted zip...")