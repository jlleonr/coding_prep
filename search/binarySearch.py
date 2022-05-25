from typing import List
'''
example:

input = [-1,0,3,5,9,12]
target = 9

[-1,0,3,5,9,12]
left = 0
right = 5
mid = 3

5 < 9 so move to right side of the array
left = mid + 1
left = 4
'''
def binarySearch(nums: List[int], left:int, right: int, target: int) -> int:

    if right >= left:

        #calculate mid
        mid = (left + right) // 2
            
        # if nums[mid] = target return mid
        if nums[mid] == target:
            return mid

        #if nums[mid] < target: ignore left part and move to the right
        if nums[mid] < target:
            #ignore left part and move to the right
            return binarySearch(nums, mid + 1, right, target)
        else:
            #ignore right part and move to the left
            return binarySearch(nums, left, mid - 1, target)

    return -1

nums = [-1,0,3,5,9,12]
result = binarySearch(nums=nums, left=0, right=len(nums)-1, target=9)

print(result)
