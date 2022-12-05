"""You are given an integer array prices where prices[i] is the price of a
given stock on the ith day.On each day, you may decide to buy and/or sell the
stock. You can only hold at most one share of the stock at any time. However,
you can buy it then immediately sell it on the same day.

Find and return the maximum profit you can achieve.

Example:
Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5),
profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Total profit is 4 + 3 = 7.
"""


from typing import List


def solution(prices: List[int]) -> int:

    length: int = len(prices)

    left: int = 0
    right: int = 1
    max_profit: int = 0

    while right < length:

        if left > prices[right]:
            left = right
            right += 1

        elif (right + 1) < length:
            if prices[right + 1] > prices[right]:
                right += 1
            else:
                max_profit += prices[left] - prices[right]
                left = right + 1
                right = left + 1
        elif prices[length - 1] > prices[left]:
            max_profit += prices[length - 1] - prices[left]

    return max_profit


# prices = [7, 1, 5, 3, 6, 4]
# prices = [1, 2, 3, 4, 5]
prices = [7, 6, 4, 3, 1]
print(solution(prices=prices))
