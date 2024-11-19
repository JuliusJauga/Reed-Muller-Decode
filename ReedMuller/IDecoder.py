from abc import ABC, abstractmethod

class IDecoder(ABC):
    @abstractmethod
    def decode(self):
        pass