from random import randint


class GridWorld:
    def __init__(self, width, height):
        """
        Initialize grid world with given width and height.

        :param width: width
        :param height: height
        """
        self.step = 0
        self.width = width
        self.height = height
        self.agent = None
        self.objects = []

    def next_step(self):
        """
        Perform next step.
        """
        self.step += 1

    def add_agent(self, agent):
        """
        Add agent.

        :param agent: agent
        """
        self.agent = agent

    def add_object(self, grid_world_object):
        """
        Add grid world object.

        :param grid_world_object: grid world object
        """
        self.objects.append(grid_world_object)

    def get_objects(self):
        """
        Return list of objects.

        :return: list of objects
        """
        return [self.agent] + self.objects if self.agent is not None else self.objects

    def get_object_types(self):
        """
        Return list of grid world object types.

        :return: list of grid world object types
        """
        return [type(grid_world_object) for grid_world_object in self.get_objects()]

    def get_objects_by_type(self, grid_world_object_type):
        """
        Return list of grid world objects of given type.

        :param grid_world_object_type: type
        :return: list of grid world objects of given type
        """
        return [grid_world_object for grid_world_object in self.get_objects()
                if type(grid_world_object) is grid_world_object_type]

    def get_object_by_type(self, grid_world_object_type):
        """
        Return grid world object of given type.

        :param grid_world_object_type: type
        :return: object of given type
        """
        objects = self.get_objects_by_type(grid_world_object_type)

        return objects[0] if len(objects) > 0 else None

    def get_free_positions(self):
        """
        Return list of free positions.

        :return: list of free positions
        """
        def is_position_free(x, y):
            for grid_world_object in self.get_objects():
                if grid_world_object.is_at(x, y):
                    return False
            return True

        positions = [(x, y) for x in range(self.width) for y in range(self.height)]

        return list(filter(lambda pos: is_position_free(*pos), positions))

    def get_random_free_position(self):
        """
        Return random free positions in the grid world where no other object is.

        :return: random free x, y position
        """
        x, y = randint(0, self.width - 1), randint(0, self.height - 1)

        free = True

        for grid_world_object in self.get_objects():
            if grid_world_object.is_at(x, y):
                free = False

        if free:
            return x, y
        else:
            return self.get_random_free_position()

    def copy(self):
        """
        Return deep copy

        :return: deep copy
        """
        copy = GridWorld(self.width, self.height)

        copy.step = self.step

        copy.add_agent(self.agent.copy())

        for grid_world_object in self.objects:
            copy.add_object(grid_world_object.copy())

        return copy


class GridWorldObject:
    def __init__(self, x, y):
        """
        Initialize grid world object with given x and y.

        :param x: x
        :param y: y
        """
        self.x = x
        self.y = y

    def is_at(self, x, y):
        """
        Return if this grid world object is at given x and y.

        :param x: x
        :param y: y
        :return:  if this grid world object is at given x and y
        """
        return self.x == x and self.y == y

    def is_at_object(self, grid_world_object):
        """
        Return if this grid world object is at position of given grid world object.

        :param grid_world_object: grid_world object
        :return: if this grid world object is at position of given grid_world object
        """
        return self.x == grid_world_object.x and self.y == grid_world_object.y

    def is_at_any_object(self, grid_world_objects):
        """
        Return if this grid world object is at position of any of given grid world objects.

        :param grid_world_objects: grid world objects
        :return: if this grid world object is at position of any of given grid world objects
        """
        for grid_world_object in grid_world_objects:
            if self.is_at_object(grid_world_object):
                return True
        return False

    def copy(self):
        """
        Return deep copy.

        :return: deep copy
        """
        return GridWorldObject(self.x, self.y)


class Agent(GridWorldObject):
    def __init__(self, x, y):
        """
        Initialize agent with given x and y.

        :param x: x
        :param y: y
        """
        super().__init__(x, y)

    def copy(self):
        return Agent(self.x, self.y)


class Treasure(GridWorldObject):
    def __init__(self, x, y):
        """
        Initialize treasure with given x and y.

        :param x: x
        :param y: y
        """
        super().__init__(x, y)

    def copy(self):
        return Treasure(self.x, self.y)


