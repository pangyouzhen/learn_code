import keyboard
import pyautogui
from loguru import logger
import time

pyautogui.hotkey("winleft", "2")
logger.info("open success")
time.sleep(5)

# keyboard.press_and_release("f5")
# logger.info("f5")
# keyboard.press_and_release("home")
# logger.info("home")
time.sleep(10)
for i in range(32):
    img = pyautogui.screenshot()
    img.save(f'./temp/{i}.jpg')
    logger.info(f"save {i} success")
    keyboard.press_and_release("pagedown")
    time.sleep(15)
