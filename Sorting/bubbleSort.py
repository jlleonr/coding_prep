from typing import List

def bubbleSort(nums:List[int]) -> List[int]:

    n = len(nums)

    if n <= 0:
        return nums

    for i in range(n - 1):
        for j in range(n - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]

    return nums

arr = [18, 2,35, 36, 15, 20, 74, 11]

print(bubbleSort(arr))