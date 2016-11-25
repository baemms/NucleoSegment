"""
Operations for layout
"""

class Layout:

    @staticmethod
    def remove_layout(layout):
        """
        Remove layout and all its widgets

        :param layout:
        :return:
        """
        if layout is not None:
            while layout.count() > 0:
                layout.takeAt(0).widget().deleteLater()

            layout.deleteLater()

    @staticmethod
    def remove_widget_from_grid(layout, row, col):
        """
        Remove widget from position

        :param row:
        :param col:
        :return:
        """
        item = layout.itemAtPosition(row, col)

        if item is not None:
            if item.widget() is not None:
                item.widget().deleteLater()

    @staticmethod
    def remove_layout_from_grid(layout, row, col):
        """
        Remove layout from position

        :param row:
        :param col:
        :return:
        """
        item = layout.itemAtPosition(row, col)

        if item is not None:
            Layout.remove_layout(item)

    @staticmethod
    def remove_parent_from_grid(layout, row, col):
        """
        Remove parent from position

        :param row:
        :param col:
        :return:
        """
        item = layout.itemAtPosition(row, col)

        if item is not None:
            item.setParent(None)
