class NoiseEnum:
    LINEAR = 0
    EASE_IN = 1
    EASE_OUT = 2
    EASE_IN_OUT = 3
    CUBIC_IN = 4
    CUBIC_OUT = 5
    CUBIC_IN_OUT = 6
    BOUNCE_OUT = 7
    ELASTIC_OUT = 8

    @classmethod
    def list_all(cls):
        return [cls.LINEAR, cls.EASE_IN, cls.EASE_OUT, cls.EASE_IN_OUT, cls.CUBIC_IN, cls.CUBIC_OUT, cls.CUBIC_IN_OUT, cls.BOUNCE_OUT, cls.ELASTIC_OUT]
    
    @staticmethod
    def to_string(noise_type):
        if noise_type == NoiseEnum.LINEAR:
            return "Linear"
        elif noise_type == NoiseEnum.EASE_IN:
            return "Ease In"
        elif noise_type == NoiseEnum.EASE_OUT:
            return "Ease Out"
        elif noise_type == NoiseEnum.EASE_IN_OUT:
            return "Ease In-Out"
        elif noise_type == NoiseEnum.CUBIC_IN:
            return "Cubic In"
        elif noise_type == NoiseEnum.CUBIC_OUT:
            return "Cubic Out"
        elif noise_type == NoiseEnum.CUBIC_IN_OUT:
            return "Cubic In-Out"
        elif noise_type == NoiseEnum.BOUNCE_OUT:
            return "Bounce Out"
        elif noise_type == NoiseEnum.ELASTIC_OUT:
            return "Elastic Out"
        else:
            raise ValueError("Invalid noise type")