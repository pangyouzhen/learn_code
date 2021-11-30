import pyautogui
from loguru import logger

# 配合快捷键使用更佳
# 按着不放 winleft 也是super键
# pyautogui.keyDown("winleft")
# 松开按键
pyautogui.keyUp("winleft")
# 直接按键
# pyautogui.press("1")

# 组合键1j

pyautogui.hotkey("winleft", "1")
# 输入文字
pyautogui.typewrite("j")
pyautogui.press("enter")
# pip  install opencv-python --timeout 10000 配合opencv进行截图
t = pyautogui.locateOnScreen("../name.png", confidence=0.5)
pyautogui.click(t)
# pyautogui.typewrite 不能直接输入中文,需要配合 pyperclip 进行使用
# 配合schedule 做定时任务
