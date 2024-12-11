import random
import math
from .NoiseEnum import NoiseEnum
from .EasingFunctions import EasingFunctions
import threading


class NoiseApplicator:
    @staticmethod
    def apply_noise(message, noise_type, noise_amount):
        if noise_amount > 1 or noise_amount < 0:
            raise ValueError("Noise amount must be between 0 and 1")
        if noise_type == NoiseEnum.LINEAR:
            return NoiseApplicator.apply_noise_with_ease(message, noise_amount, EasingFunctions.linear)
        elif noise_type == NoiseEnum.EASE_IN:
            return NoiseApplicator.apply_noise_with_ease(message, noise_amount, EasingFunctions.ease_in)
        elif noise_type == NoiseEnum.EASE_OUT:
            return NoiseApplicator.apply_noise_with_ease(message, noise_amount, EasingFunctions.ease_out)
        elif noise_type == NoiseEnum.EASE_IN_OUT:
            return NoiseApplicator.apply_noise_with_ease(message, noise_amount, EasingFunctions.ease_in_out)
        elif noise_type == NoiseEnum.CUBIC_IN:
            return NoiseApplicator.apply_noise_with_ease(message, noise_amount, EasingFunctions.cubic_in)
        elif noise_type == NoiseEnum.CUBIC_OUT:
            return NoiseApplicator.apply_noise_with_ease(message, noise_amount, EasingFunctions.cubic_out)
        elif noise_type == NoiseEnum.CUBIC_IN_OUT:
            return NoiseApplicator.apply_noise_with_ease(message, noise_amount, EasingFunctions.cubic_in_out)
        elif noise_type == NoiseEnum.BOUNCE_OUT:
            return NoiseApplicator.apply_noise_with_ease(message, noise_amount, EasingFunctions.bounce_out)
        elif noise_type == NoiseEnum.ELASTIC_OUT:
            return NoiseApplicator.apply_noise_with_ease(message, noise_amount, EasingFunctions.elastic_out)
        else:
            raise ValueError("Invalid noise type")
    
    @staticmethod
    def apply_noise_with_ease(message, noise_amount, easing_function):
        if not 0 <= noise_amount <= 1:
            raise ValueError("Noise amount must be between 0 and 1")

        message_length = len(message)
        if message_length == 0:
            return message, []
        if message_length > 16*16*8*3:
            return NoiseApplicator.apply_noise_sequentially(message, noise_amount, easing_function)

        noisy_message = message.copy()
        mistake_positions = []

        for i, bit in enumerate(message):
            normalized_position = i / (message_length - 1) if message_length > 1 else 0
            adjusted_noise_amount = noise_amount * easing_function(normalized_position)

            if random.random() < adjusted_noise_amount:
                noisy_message[i] = 1 - bit
                mistake_positions.append(i)

        return noisy_message, mistake_positions
    def apply_noise_sequentially(message, noise_amount, easing_function):
        # print("Applying noise sequentially")
        if not 0 <= noise_amount <= 1:
            raise ValueError("Noise amount must be between 0 and 1")

        message_length = len(message)
        if message_length == 0:
            return message, []

        noisy_message = message.copy()
        mistake_positions = []
        lock = threading.Lock()

        def apply_noise_to_segment(start, end):
            local_mistake_positions = []
            for i in range(start, end):
                normalized_position = i / (message_length - 1) if message_length > 1 else 0
                adjusted_noise_amount = noise_amount * easing_function(normalized_position)

                if random.random() < adjusted_noise_amount:
                    with lock:
                        noisy_message[i] = 1 - message[i]
                        local_mistake_positions.append(i)
            with lock:
                mistake_positions.extend(local_mistake_positions)

        num_threads = 4
        segment_length = math.ceil(message_length / num_threads)
        threads = []

        for i in range(num_threads):
            start = i * segment_length
            end = min((i + 1) * segment_length, message_length)
            thread = threading.Thread(target=apply_noise_to_segment, args=(start, end))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return noisy_message, mistake_positions