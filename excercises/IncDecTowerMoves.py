from typing import List


'''
For an increasing or decreasing tower, give the least number of moves needed to
complete the tower.

Example.
    Input = [1, 4, 3, 2]
    Output = 4

    Explanation:    We add 4 to 1 to have [5, 4, 3, 2]
'''

'''
Logic:

- Check if elements are in increasing order
    - For every element
        - Check that n: n < n + 1 = True
            - If not:
                While n > n + 1
                    do n -= 1
                    count += 1
                return count

- If not in increasing order
    - Elements must be in decreasing order
        - For every element
        - Check that n: n > n + 1 = True
            - If not:
                While n < n + 1
                    do n += 1
                    count += 1
                return count
'''


def moves_needed(list: List[int]) -> int:
    # Get the length of the array
    n: int = len(list)

    # Check if elements are in increasing order
    if (list[0] <= list[1]) and (list[n - 2] <= list[n - 1]):
        make_increasing(list)
    # Check if elements are in decreasing order
    elif (list[0] >= list[1]) and (list[n - 2] >= list[n - 1]):
        pass  # decreasing TODO
    # Check if increasing and then decreasing
    elif (list[0] <= list[1]) and (list[n - 2] >= list[n - 1]):
        pass  # increasing then decreasing
    else:
        pass  # decreasing then increasing TODO


def make_decreasing(list: List[int]) -> int:
    count: int = 0
    for i in range(len(list) - 1):

        while list[i] <= list[i + 1]:
            list[i] += 1
            count += 1

    return count


def make_increasing(list: List[int]) -> int:
    count: int = 0
    for i in range(len(list) - 1):

        while list[i] >= list[i + 1]:
            list[i+1] += 1
            count += 1

    return count


lst = [1, 4, 2, 2]
print(make_decreasing(lst))

lst = [2, 3, 4, 1]
print(make_increasing(lst))