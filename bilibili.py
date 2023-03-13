import requests
from lxml import etree
import re
import json
import os
import moviepy.editor as mpy
import argparse
import multiprocessing
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36', 
    "referer": "https://message.bilibili.com/"
}

def get_series_bv(url):
    mid = re.search("(?<=com/).*?(?=/)", url).group()
    sid = re.search("(?<=sid=).*?(?=&)", url).group()
    document_list = os.listdir("./")
    
    if mid + "_" + sid + ".txt"  not in document_list:       
        print("not find archive cache")
        archive = "https://api.bilibili.com/x/series/archives?mid=" + mid + "&series_id=" + sid + "&pn=1&ps=10000"
        try:
            response = requests.get(archive, headers=headers)
            response.raise_for_status()
            with open(mid + "_" + sid + ".txt", "wb") as f:
                f.write(response.content)
            response = response.content
        except:
            print("get archive from " + url + " failed")
            return None, None
    else:
        print("find archive cache")
        with open(mid + "_" + sid + ".txt", "r") as f:
            response = f.read()

    bv_json = json.loads(response)["data"]["archives"]
    url_list = []
    bv_list = []
    for video in bv_json:
        url_list.append("https://www.bilibili.com/video/" + video["bvid"])
        bv_list.append(video["bvid"])
    return url_list, bv_list
    

def get_audio_url(url):
    session = requests.session()
    response = session.get(url, headers=headers)
    try:
        response.raise_for_status()
    except:
        print("get audio url from " + url + " failed")
        return

    html = etree.HTML(response.content)
    video_infos = html.xpath('//head/script[3]')[0].text
    url = re.findall("Url\":\".*?\"", video_infos)
    url = [u[6:-2] for u in url]
    return url[-1]


def get_audio_data(url, name, output_path):
    try:
        res = requests.get(url, headers=headers).content
    except:
        print(f"download audio {name} failed")
        return False
    
    with open(os.path.join(output_path, name + ".mp4"), "wb") as f:
        f.write(res)
    
    return True


def mp4_to_mp3(path):
    pool_sema.acquire()
    print("convert " + path + " to MP3...")
    audio = mpy.AudioFileClip(path) 
    audio_path = path[:-1] + '3'
    audio.write_audiofile(audio_path, logger=None)
    os.remove(path)
    print(audio_path + " have been convert")
    pool_sema.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download from bilibili archive")
    parser.add_argument("-u", "--url", required=True, type=str, help="bilibili seriesdetail url")
    parser.add_argument("-o", "--output_dir", default="./", type=str, help="path to output audio")
    parser.add_argument("-j", "--jobs", default=2, type=int, help="the number of multithreading allowed")
    args = parser.parse_args().__dict__
    archive = args.pop("url")
    # archive = "https://space.bilibili.com/667526012/channel/seriesdetail?sid=210559&ctype=0"
    # archive = "https://space.bilibili.com/1660392980/channel/seriesdetail?sid=1961996&ctype=0"
    output_path = os.path.join(args.pop("output_dir"), "RAW")
    os.makedirs(name=output_path, exist_ok=True)

    jobs = args.pop("jobs")
    pool_sema = multiprocessing.BoundedSemaphore(jobs)
    url_list, bv_list = get_series_bv(archive)
    if url_list is None:
        print("fail to find bv url")
        exit(0)
    print("get video url!")
    
    file_list = os.listdir(output_path)
    
    pool = multiprocessing.Pool(jobs)
    for url, bv in zip(url_list, bv_list):
        audio_url = get_audio_url(url)
        if bv + ".mp4" in file_list:
            print(bv + " have been download")
            pool.apply_async(mp4_to_mp3, args=[os.path.join(output_path, bv + ".mp4")])
            continue

        if bv + ".mp3" in file_list:
            print(bv + " have been download")
            continue

        print("try to download in " + url)
        status = get_audio_data(audio_url, bv, output_path)

        if status:
            pool.apply_async(mp4_to_mp3, args=[os.path.join(output_path, bv + ".mp4")])
    pool.close()
    pool.join()

    print("done")