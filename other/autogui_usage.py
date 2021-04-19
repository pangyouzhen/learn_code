import pyautogui

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
