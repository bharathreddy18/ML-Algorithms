from abc import ABC, abstractmethod

# Create an abstract class
class DiningExperience(ABC):
    # Template method
    def serve_dinner(self):
        self.serve_appetizer()
        self.serve_main_course()
        self.serve_dessert()
        self.serve_beverage()

    @abstractmethod
    def serve_appetizer(self):
        pass

    @abstractmethod
    def serve_main_course(self):
        pass

    @abstractmethod
    def serve_dessert(self):
        pass

    @abstractmethod
    def serve_beverage(self):
        pass

# Create concrete class
class ItalianDinner(DiningExperience):
    def serve_appetizer(self):
        print('Serving bruschetta as appetizer.')

    def serve_main_course(self):
        print('Serving pasta as main course.')

    def serve_dessert(self):
        print('Serving tirimasu as dessert.')

    def serve_beverage(self):
        print('Serving wine as a beverage.')

class ChineseDinner(DiningExperience):
    def serve_appetizer(self):
        print('Serving Spring rolls  as appetizer.')

    def serve_main_course(self):
        print('Serving stir fried noodles as main course.')

    def serve_dessert(self):
        print('Serving cookies as dessert.')

    def serve_beverage(self):
        print('Serving tea as a beverage.')

# Client code
if __name__ == "__main__":
    print('Italian Dinner: ')
    italian = ItalianDinner()
    italian.serve_dinner()

    print('\nChinese Dinner: ')
    chinese = ChineseDinner()
    chinese.serve_dinner()