import math

# Class to provide easing functions for noise application
class EasingFunctions:
    @staticmethod
    def linear(t):
        """Linear easing (no acceleration)."""
        return 1

    @staticmethod
    def ease_in(t):
        """Quadratic ease-in: starts slow, accelerates."""
        return t ** 2

    @staticmethod
    def ease_out(t):
        """Quadratic ease-out: starts fast, decelerates."""
        return (1 - t) ** 2

    @staticmethod
    def ease_in_out(t):
        """Quadratic ease-in-out: combines ease-in and ease-out."""
        if t < 0.5:
            return 4 * (t ** 2)
        return 4 * ((1 - t) ** 2)

    @staticmethod
    def cubic_in(t):
        """Cubic ease-in: starts even slower than quadratic."""
        return t ** 3

    @staticmethod
    def cubic_out(t):
        """Cubic ease-out: ends slower than quadratic."""
        return 1 - (1 + (t - 1) **3)

    @staticmethod
    def cubic_in_out(t):
        """Cubic ease-in-out: smoother than quadratic."""
        if t < 0.5:
            return 8 * t ** 3
        return 8 * ((1 - t) ** 3)

    @staticmethod
    def bounce_out(t):
        """Bounce ease-out: simulates a bounce effect."""
        if t < 1 / 2.75:
            return 7.5625 * t * t
        elif t < 2 / 2.75:
            t -= 1.5 / 2.75
            return 7.5625 * t * t + 0.75
        elif t < 2.5 / 2.75:
            t -= 2.25 / 2.75
            return 7.5625 * t * t + 0.9375
        else:
            t -= 2.625 / 2.75
            return 7.5625 * t * t + 0.984375

    @staticmethod
    def elastic_out(t):
        """Elastic ease-out: simulates a spring effect."""
        if t == 0 or t == 1:
            return t
        return math.pow(2, -10 * t) * math.sin((t - 0.075) * (2 * math.pi) / 0.3) + 1
    