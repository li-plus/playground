from abc import ABC, abstractmethod


class Animal(ABC):
    @abstractmethod
    def run(self):
        pass


class Dog(Animal):
    def run(self):
        print("dog is running")


class Cat(Animal):
    def run(self):
        print("cat is running")


try:
    a = Animal()
except Exception as e:
    print(e)

dog = Dog()
dog.run()

cat = Cat()
cat.run()
