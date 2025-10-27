from abc import ABC, abstractmethod

# Define the strategy interface
class PaymentMethod(ABC):
    @abstractmethod
    def pay(self, amount):
        pass

# Implement concrete strategies
class CreditCardPayment(PaymentMethod):
    def pay(self, amount):
        return f"Paying {amount} using credit card."
    
class PayPalPayment(PaymentMethod):
    def pay(self, amount):
        return f"Paying {amount} using paypal."
    
class BitcoinPayment(PaymentMethod):
    def pay(self, amount):
        return f'Paying {amount} using Bitcoin.'
    
# Implement the context
class ShoppingCart:
    def __init__(self, payment_method: PaymentMethod):
        self.payment_method = payment_method
    
    def checkout(self, amount):
        return self.payment_method.pay(amount)
    
# Use the strategy in the context
if __name__ == "__main__":
    cart = ShoppingCart(CreditCardPayment())
    print(cart.checkout(1000))

    cart = ShoppingCart(PayPalPayment())
    print(cart.checkout(2000))

    cart = ShoppingCart(BitcoinPayment())
    print(cart.checkout(3000))