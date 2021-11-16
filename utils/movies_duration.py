from pathlib import Path

from moviepy.video.io.VideoFileClip import VideoFileClip

total_seconds = 0
path = Path("D:\\project\\资料\\玩转数据结构 从入门到进阶")
for i in path.glob("*/*mp4"):
    # print(str(i.absolute()))
    s = str(i.absolute())
    clip = VideoFileClip(s)
    total_seconds += clip.duration
    print("--------------")
print(total_seconds)
