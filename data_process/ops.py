# coding=utf-8


def find_local_minimum(data):
    """
    Find All Local minimums in a list of Integers.
    An element is a local minimum if it is larger than the two elements adjacent to it,
    or if it is the first or last element and larger than the one element adjacent to it.
    Design an algorithm to find all local minimum if they exist.
    Example:  for 3, 2, 4, 1 the local minimum are at indices 1 and 3.
    :param data: 1-D sequence.
    :return: minimum list, index list
    """
    minimum = []
    index = []
    length = len(data)
    if length >= 2:
        if data[0] < data[1]:
            minimum.append(data[0])
            index.append(0)

        if length > 3:
            for i in range(1, length - 1):
                if data[i] < data[i - 1] and data[i] < data[i + 1]:
                    minimum.append(data[i])
                    index.append(i)

        if data[length - 1] < data[length - 2]:
            minimum.append(data[length - 1])
            index.append(length - 1)
    return minimum, index


def find_local_maximum(data):
    """
    Find All Local Maximums in a list of Integers.
    An element is a local maximum if it is larger than the two elements adjacent to it,
    or if it is the first or last element and larger than the one element adjacent to it.
    Design an algorithm to find all local maximum if they exist.
    Example:  for 3, 2, 4, 1 the local maximum are at indices 0 and 2.
    :param data: 1-D sequence.
    :return: maximum list, index list
    """
    maximum = []
    index = []
    length = len(data)
    if length >= 2:
        if data[0] > data[1]:
            maximum.append(data[0])
            index.append(0)

        if length > 3:
            for i in range(1, length - 1):
                if data[i] > data[i - 1] and data[i] > data[i + 1]:
                    maximum.append(data[i])
                    index.append(i)

        if data[length - 1] > data[length - 2]:
            maximum.append(data[length - 1])
            index.append(length - 1)
    return maximum, index


def main():
    numbers = [3, 2, 4, 1, 7]
    num, idx = find_local_minimum(numbers)
    print(idx)
    print(num)


if __name__ == "__main__":
    main()
