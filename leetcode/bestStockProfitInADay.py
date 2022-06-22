"""You are given an integer array prices where prices[i] is the price
of a given stock during a day at a certain hour.

Find and return the maximum profit you can achieve.

Example 1.
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on hour #2 (price = 1) and sell on hour #4 (price = 6),
profit = 6-1 = 5.

Example 2:

Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on hour #1 (price = 1) and sell on hour #5 (price = 5),
profit = 5-1 = 4.

Total profit is 4.
"""

from typing import List

"""
Solution:
Initialize buy to first price. Then if sell price is less than buy price,
update sell price = buy price and calculate max profit. Move sell price.
"""


def solution(input: List[int]) -> int:
    """Return max profit from selling stocks in a day

    Args:
        input (List[int]): Stock prices in a day

    Returns:
        int: Max profit
    """

    left: int = prices[0]
    right: int = 1
    maxProfit: int = 0

    while right < len(prices):

        if prices[right] < left:
            left = prices[right]
        else:
            maxProfit = max(maxProfit, prices[right] - left)

        right += 1

    return maxProfit

# Test cases:
# prices = [7, 1, 5, 3, 6, 4]
# prices = [1, 2, 3, 4, 5]
# prices = [3, 2, 6, 5, 0, 3]


prices = [2, 1, 2, 1, 0, 1, 2]
print(solution(prices))
