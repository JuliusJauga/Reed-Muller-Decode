import random
import math
from .NoiseEnum import NoiseEnum
from .EasingFunctions import EasingFunctions


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
        noisy_message = ""
        mistake_positions = []
        message_length = len(message)

        for i, bit in enumerate(message):
            # Normalize position (0 to 1 for the message length)
            normalized_position = i / (message_length - 1) if message_length > 1 else 0

            # Adjust noise probability using the easing function
            adjusted_noise_amount = noise_amount * easing_function(normalized_position)

            # Apply noise based on the adjusted probability
            if random.random() < adjusted_noise_amount:
                # Flip the bit
                noisy_message += str(int(not int(bit)))
                mistake_positions.append(i)  # Store the original position of the mistake
            else:
                # Keep the original bit
                noisy_message += bit

        return noisy_message, mistake_positions
