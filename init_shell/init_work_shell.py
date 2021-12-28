import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--type", help="git bash/mobaxterm/linuxshell")
parser.add_argument("--home")
args = parser.parse_args()


class InitShell:
    def __init__(self):
        pass

    def init_shell(self):
        pass

    def plugin(self):
        raise NotImplementedError

    def common_handle(self):
        pass


class WindowsShell:
    def __init__(self):
        pass

    def init_shell(self):
        pass

    def plugin(self):
        raise NotImplementedError

    def common_handle(self):
        pass

class GitBashShell(WindowsShell):
    def __init__(self):
        super().__init__()
        self.path = None

    def plugin(self):
        pass

class MobaxtermShell(WindowsShell):
    def __init__(self):
        super().__init__()
        self.path = None

    def plugin(self):
        pass

class LinuxShell(InitShell):
    def __init__(self):
        super().__init__()
        self.path = None

    def plugin(self):
        pass
