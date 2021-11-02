import time
import types

import pyautogui


# todo not work
# https://stackoverflow.com/questions/8951787/defining-python-decorators-for-a-complete-module
def decorator(func):
    def wrapper(*args, **kwargs):
        # a = func(*args, **kwargs)
        time.sleep(3)
        print("time sleep 3")
        # print(a)
        # return a

    return wrapper


for k, v in vars(pyautogui).items():
    if isinstance(v, types.FunctionType):
        vars(pyautogui)[k] = decorator(v)

# pyautogui.hotkey("winleft", "1")
