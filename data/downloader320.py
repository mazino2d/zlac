import requests
import wget
import pandas as pd
import argparse
import time
from os import listdir, mkdir, system


# Add the arguments to the parser
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--count", required=True)
args = vars(ap.parse_args())
COUNT = int(args['count'])

# read file to get song id
song_info = pd.read_csv('./trend_song.csv')
song_id = song_info['id']

exist_ids = []

try:
    exist_ids = [int(f[:-4]) for f in listdir("./origin/320")]
except FileNotFoundError:
    print("[WARNING] No 'data' directory")
    mkdir('./origin/320')

song_id = [x for x in song_id if x not in exist_ids]

# api-endpoint (this is private link)
URL = "https://zmediadata.zingmp3.vn/api/song/mGetInfoMedia?" + \
    "infoSrc=webZMD&typeLink=audio_320&listKey=%s"

count = 1
for id in song_id:
    # sending get request and saving the response as response object
    r = requests.get(url=URL % (id))
    # extracting data in json format
    link = r.json()[0]['link']
    # download mp3 file
    wget.download(link, 'origin/320/%s.mp3' % (id))

    if count == COUNT:
        break
    else:
        count = count + 1
