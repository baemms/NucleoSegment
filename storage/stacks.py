"""
Struct class to enable to pass on stack data
"""


class Stack:

    def copy(self):
        """
        Return copy of all the elements set

        :return:
        """
        stack = Stack()

        # add variables to new stack
        for var in self.__dict__:
            stack.__setattr__(var, self.__getattribute__(var))

        return stack

