import keyboard
import pyautogui
from loguru import logger
import time

pyautogui.hotkey("winleft", "4")
logger.info("open success")
time.sleep(5)

keyboard.press_and_release("f5")
logger.info("f5")
time.sleep(10)
for i in range(200):
    img = pyautogui.screenshot()
    img.save(f'./ld/{i}.jpg')
    logger.info(f"save {i} success")
    keyboard.press_and_release("pagedown")
    time.sleep(10)
