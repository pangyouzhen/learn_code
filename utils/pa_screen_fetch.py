import time
from math import ceil
from pathlib import Path
from typing import List

import keyboard
import pyautogui
from loguru import logger


class Context:
    def __init__(self, screen):
        self.screen = screen

    def save(self, pages: List[int]):
        if isinstance(pages, int):
            pages = [pages]
        return self.screen.run(pages)


class Screen:
    def run(self, pages: List[int]) -> None:
        self.init_dcloud()
        pages = self.convert_pages(pages)
        for i in range(len(pages)):
            path = Path(f"./temp/{i}")
            path.mkdir(exist_ok=True)
            self.switch()
            self.save(pages, i)

    def switch(self):
        assert NotImplementedError

    def save(self, pages, i) -> None:
        for j in range(pages[i]):
            img = pyautogui.screenshot()
            img.save(f'./temp/{i}/{j}.jpg')
            logger.info(f"save {j} success")
            keyboard.press_and_release("pagedown")
            time.sleep(3)

    def convert_pages(self, pages: List[int]) -> List[int]:
        return pages

    def init_dcloud(self):
        keyboard.press_and_release('left windows + 2')
        logger.info("open success")
        time.sleep(5)


class Txt(Screen):

    def convert_pages(self, pages: List[int]):
        pages = [ceil((i / 49)) for i in pages]
        return pages

    def switch(self):
        logger.info("文本类型-进入切换")
        time.sleep(3)
        ctrl_tab()
        time.sleep(4)


class HorizonalPPT(Screen):
    def switch(self):
        logger.info("横屏ppt-进入下一个")
        keyboard.press_and_release("esc")
        time.sleep(4)
        # 切换下一个pdf
        ctrl_tab()
        time.sleep(4)
        keyboard.press_and_release("f5")
        time.sleep(6)


class VertPPT(Screen):
    def switch(self):
        logger.info("竖屏pdf--进入下一个pdf")
        # 退出pdf全屏
        keyboard.press_and_release("esc")
        time.sleep(4)
        # 切换下一个pdf
        ctrl_tab()
        time.sleep(4)
        # 顺时针旋转
        keyboard.press_and_release("ctrl+shift+=")
        time.sleep(4)
        # 进入全屏
        keyboard.press_and_release("ctrl+l")
        time.sleep(6)


def ctrl_tab():
    keyboard.press_and_release("ctrl+tab")
    time.sleep(2)


if __name__ == '__main__':
    pages = [6]
    strategy = Context(HorizonalPPT())
    strategy.save(pages)
