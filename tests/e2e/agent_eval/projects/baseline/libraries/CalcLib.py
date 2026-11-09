class CalcLib:
    """Tiny calculator library used by the lab."""

    def add_numbers(self, a, b):
        """Adds two numbers and returns the result."""
        return float(a) + float(b)

    def apply_discount(self, price, percent):
        """Returns price reduced by percent. (BUG: subtracts percent as absolute value)"""
        return float(price) - float(percent)

    def format_money(self, amount, currency="EUR"):
        """Formats amount with 2 decimals followed by currency."""
        return f"{float(amount):.2f} {currency}"
