import csv

input_csv = "csv/2020.csv"      # 原始只有 video_id 列
output_csv = "csv/2020_2.csv"   # 输出带 filepath 的新文件

with open(input_csv, "r", newline="") as f_in, \
     open(output_csv, "w", newline="") as f_out:

    reader = csv.reader(f_in)
    writer = csv.writer(f_out)
    
    # 写入表头
    writer.writerow(["video_id", "filepath"])
    
    for row in reader:
        video_id = row[0].strip()
        filepath = f"2020/{video_id}.mkv"
        writer.writerow([video_id, filepath])

print(f"Done! Output saved to {output_csv}")