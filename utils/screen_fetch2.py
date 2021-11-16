import time
from math import ceil
from pathlib import Path

import keyboard
import pyautogui
from loguru import logger

pyautogui.hotkey("winleft", "2")
logger.info("open success")
time.sleep(5)

all_ = [73, 130, 28, 1832]
_ = [ceil((i / 49)) for i in all_]
print(_)
for i in range(len(all_)):
    path = Path(f"./cv/{i}")
    path.mkdir(exist_ok=True)
    if i != 0:
        keyboard.press("ctrl")
        time.sleep(2)
        keyboard.press_and_release("tab")
        time.sleep(2)
        keyboard.release("ctrl")
        time.sleep(2)
    keyboard.press_and_release("home")
    logger.info("home")
    for j in range(_[i]):
        img = pyautogui.screenshot()
        img.save(f'./cv/{i}/{j}.jpg')
        logger.info(f"save {j} success")
        keyboard.press_and_release("pagedown")
        time.sleep(3)
    time.sleep(5)
    logger.info("进入下一个")
