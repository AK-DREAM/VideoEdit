import csv
import os, shutil
from os.path import join as osj

input_csv = "csv/mashup_2.csv"      # 原始只有 video_id 列
output_csv = "csv/mashup.csv"   # 输出带 filepath 的新文件

with open(input_csv, "r", newline="") as f_in, \
     open(output_csv, "w", newline="") as f_out:

    reader = csv.reader(f_in)
    writer = csv.writer(f_out)
    
    # 写入表头
    writer.writerow(["video_id", "filepath"])
    
    base_dir = "../data/videos"

    for row in reader:
        video_id = row[0].strip()
        # from_filepath = f"mashup/{video_id}.mp4"
        to_filepath = f"mashup/{video_id}/video.mp4"
        # if not os.path.exists(osj(base_dir, from_filepath)):
            # continue
        # os.mkdir(osj(base_dir, "mashup", video_id))
        # shutil.move(osj(base_dir, from_filepath), osj(base_dir, to_filepath))
        writer.writerow([video_id, to_filepath])

print(f"Done! Output saved to {output_csv}")