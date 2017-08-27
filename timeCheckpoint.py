from time import clock

def timeCheckpoint(start_time, name):
    """Used for timing code. Prints the time since start_time in seconds, along with the given name.

    :return: the current time
    """

    time = clock() - start_time
    print(str.capitalize(name) + ': \t%.3f' % time)
    return clock()

checkpoint = clock()