from env.tasks.find_treasure import FindTreasureTask

__all__ = [
    'FindTreasureTask'
]

# Predefined tasks
predefined = {
    "find_treasure_v0": FindTreasureTask(width=9, height=9, episode_length=60, treasure_position=(6, 7)),
    "find_treasure_v1": FindTreasureTask(width=9, height=9, episode_length=60)
}


def find_task(task_name):
    """
    Find given task in predefined tasks.

    :param task_name: task name
    :return: task
    """
    if task_name not in predefined:
        raise ValueError("Unknown task: " + task_name)

    return predefined[task_name]
