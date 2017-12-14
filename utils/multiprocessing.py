def serialize(obj):
    """
    Serialize object using cloudpickle.

    :param obj: object
    :return: serialized object
    """
    import cloudpickle
    return cloudpickle.dumps(obj)


def deserialize(obj):
    """
    Deserialize object.

    :param obj: object
    :return: deserialized object
    """
    import pickle
    return pickle.loads(obj)
