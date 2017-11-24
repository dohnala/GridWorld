from kivy.app import App
from kivy.graphics import Rectangle
from kivy.uix.widget import Widget


class GridWorldGUI(Widget):
    def __init__(self):
        super(GridWorldGUI, self).__init__()
        with self.canvas:
            Rectangle(pos=(self.center_x, self.center_y))


class GridWorldApp(App):
    def build(self):
        return GridWorldGUI()


if __name__ == '__main__':
    GridWorldApp().run()
