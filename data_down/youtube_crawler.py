import subprocess
import argparse
import os
from pathlib import Path

def get_ids_from_playlist(playlist_url):
    """返回 playlist 中所有 video id"""
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--get-id",
        playlist_url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    ids = result.stdout.strip().split("\n")
    return ids

def get_ids_from_channel(channel_url):
    """返回频道上传的所有视频 id"""
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--get-id",
        channel_url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    ids = result.stdout.strip().split("\n")
    return ids

def get_ids(playlist_urls, channel_url, output_file):
    all_ids = set()

    # 两个 playlist
    for url in playlist_urls:
        print(f"Fetching from playlist: {url}")
        ids = get_ids_from_playlist(url)
        print(f"  Got {len(ids)} IDs")
        all_ids.update(ids)

    # 一个 channel
    if channel_url:
        print(f"Fetching from channel: {channel_url}")
        ids = get_ids_from_channel(channel_url)
        print(f"  Got {len(ids)} IDs")
        all_ids.update(ids)

    # 写到文件
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for vid in sorted(all_ids):
            f.write(vid + "\n")

    print(f"Done! {len(all_ids)} unique video IDs saved to {output_file}")

def youtube_download(video_ids, video_dir):
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    output_fmt = os.path.join(".", '%(id)s.%(ext)s')
    id_fp = str(video_ids)
    cmd = 'yt-dlp --config-location youtube-dl.conf -o "{}" -a "{}"'.format(output_fmt, id_fp)
    os.system(cmd)

    print("DOWNLOAD COMPLETE")

    check_missing_vids(video_ids, video_dir)

def check_missing_vids(video_ids, video_dir, video_ext='.mp4'):
    missing_ids = []
    total_ids = 0
    with open(str('video_ids.txt'), "r", encoding="utf-8") as f:
        for line in f:
            total_ids += 1
            video_id = line.strip() 
            video_fp = os.path.join(video_dir, video_id + video_ext)
            if not os.path.isfile(video_fp):
                missing_ids.append(video_id)

    success = (1 - len(missing_ids) / total_ids) * 100
    print('=======================================================\n%.2f %% of clips downloaded successfully' % success)
    if success == 100:
        pass
    elif success < 100:
        print(
            '%d clips failed to download.' % len(missing_ids))

    with open('missing_videos.txt', 'w') as fid:
        for video_id in missing_ids:
            fid.write(video_id + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--playlist_urls", nargs=2, required=True, help="Two YouTube playlist URLs")
    # parser.add_argument("--channel_url", help="YouTube channel URL")
    # parser.add_argument("--output", default="video_ids.txt", help="Output text file")
    
    parser.add_argument("--list", default="video_ids.txt", help="Output text file")
    args = parser.parse_args()
    # get_ids(args.playlist_urls, args.channel_url, args.output)
    youtube_download(str(args.list), "../data/videos/")
