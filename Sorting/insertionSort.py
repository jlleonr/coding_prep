from typing import List

def insertionSort(list: List[int]) -> List[int]:
    n: int = len(list)

    # Iterate through list starting from seconde element
    for i in range(1, n):

        #Grab right element
        tobe_sorted: int = list[i]

        # index to the left of tobe_sorted
        j = i - 1

        # While left element > right element
        while j >= 0 and list[j] > tobe_sorted:

            # shift left element to the right and decrease left index
            list[j + 1] = list[j]
            j -= 1

        # else put right element at left index + 1
        list[j + 1] = tobe_sorted

    return list



array: List = [8, 2, 6, 4, 5]

print(insertionSort(list=array))
