from abc import ABC, abstractmethod

# Interface for the decoder
class IDecoder(ABC):
    @abstractmethod
    def decode(self):
        pass