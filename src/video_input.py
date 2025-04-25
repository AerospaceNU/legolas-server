from abc import ABC, abstractmethod


class VideoInput(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def get_frame(self):
        pass

    @abstractmethod
    def shutdown(self):
        pass
