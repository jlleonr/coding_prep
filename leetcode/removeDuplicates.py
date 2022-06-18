"""
Remove duplicates from a sorted array in non-decreasing order.
Modify the array so non-repeated elements are at the begining and
return the index of the last non-repeating element.
E.g.:
Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
"""

from typing import List


class Solution:
    """This problem is solved by using two pointers.
        left pointer keeps track of the position of
        non-repeated elements and serves as the index of the last element
        that's part of the solution.
        right pointer is used to traverse looking for non-repeated
        elements.
    """
    def removeDuplicates(self, nums: List[int]) -> int:
        """Remove duplicates from a sorted array

        Args:
            nums (List[int]): Sorted non-decreasing elements

        Returns:
            int: index of past the last non-repeated element
        """

        if len(nums) < 1:
            return 1

        left = 0
        right = 1

        while right < len(nums):

            if nums[right] != nums[left]:
                left += 1
                nums[left] = nums[right]
            right += 1

        # We want to point after the last non-repeated
        # element
        return left + 1


nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
sol = Solution()
res = sol.removeDuplicates(nums)
print(res)
print(nums)