"""
Struct class to enable to pass on image data
"""


class Image:

    def copy(self):
        """
        Return copy of all the elements set

        :return:
        """
        image = Image()

        # add variables to new image
        for var in self.__dict__:
            image.__setattr__(var, self.__getattribute__(var))

        return image

