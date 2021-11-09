import time

import keyboard
import pyautogui
from loguru import logger
from pathlib import Path

pyautogui.hotkey("winleft", "2")
logger.info("open success")
time.sleep(5)

all_ = [1, 49, 2, 3, 8, 7, 7, 15, 148, 3, 1, 3, 81, 4, 3, 8, 14, 4]
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
    for j in range(all_[i]):
        img = pyautogui.screenshot()
        img.save(f'./cv/{i}/{j}.jpg')
        logger.info(f"save {j} success")
        keyboard.press_and_release("pagedown")
        time.sleep(3)
    time.sleep(5)
