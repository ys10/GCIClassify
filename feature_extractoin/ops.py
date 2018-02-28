# coding=utf-8


def find_local_minimum(data):
    minimum = []
    index = []
    length = len(data)
    if length >= 2:
        if data[0] > data[1]:
            minimum.append(data[0])
            index.append(0)

        if length > 3:
            for i in range(1, length - 1):
                if data[i] > data[i - 1] and data[i] > data[i + 1]:
                    minimum.append(data[i])
                    index.append(i)

        if data[length - 1] > data[length - 2]:
            minimum.append(data[length - 1])
            index.append(length - 1)
    return minimum, index


def find_local_maximum(data):
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
    num, idx = find_local_maximum(numbers)
    print(idx)
    print(num)


if __name__ == "__main__":
    main()
