"""
You are given coins of different denominations and a total amount of
money amount. Write a function to compute the fewest number of coins
that you need to make up that amount. If that amount of money cannot
be made up by any combination of the coins, return -1.

Example 1:

Input: coins = [1, 2, 5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
Example 2:

Input: coins = [2], amount = 3
Output: -1
"""
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        # if amount <= 0:
        #    return -1
        return self.coin_change(coins, amount, {})
    def coin_change(self, coins, remainder, memo):
        if remainder == 0:
            return 0
        if len(coins) == 0:
            return -1
        if remainder<0:
            return -1
        min_value = float('inf')
        for c in coins:
            if remainder-c not in memo:
                memo[remainder-c] = self.coin_change(coins, remainder-c, memo)
            res = memo[remainder-c]
            if res>=0 and res < min_value:
                min_value = 1+res
        if min_value<float('inf'):
            memo[remainder] = min_value
        else:
            memo[remainder] = -1
        return memo[remainder]
# Solution2, based on stack:
class Solution(object):
    def coinChange(self, coins, amount):
        if amount <0 or not coins:
            return -1
        if amount==0:
            return 0
        stack, level, visit = [0], 0, set()
        while stack:
            new_level = []
            level += 1
            for current_amount in stack:
                for c in coins:
                    if current_amount + c ==amount:
                        return level
                    if current_amount + c not in visit:
                        new_level.append(current_amount+c)
                        visit.add(current_amount+c)
            if min(new_level) > amount:
                return -1
            stack = new_level


