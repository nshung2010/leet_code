"""
375. Guess Number Higher or Lower II
Medium

479

689

Favorite

Share
We are playing the Guess Game. The game is as follows:

I pick a number from 1 to n. You have to guess which number I picked.

Every time you guess wrong, I'll tell you whether the number I picked
is higher or lower.

However, when you guess a particular number x, and you guess wrong,
you pay $x. You win the game when you guess the number I picked.

Example:

n = 10, I pick 8.

First round:  You guess 5, I tell you that it's higher. You pay $5.
Second round: You guess 7, I tell you that it's higher. You pay $7.
Third round:  You guess 9, I tell you that it's lower. You pay $9.

Game over. 8 is the number I picked.

You end up paying $5 + $7 + $9 = $21.
Given a particular n ≥ 1, find out how much money you need to have to g
uarantee a win.
"""
class Solution(object):
    def getMoneyAmount(self, n):
        """
        :type n: int
        :rtype: int
        """

        def helper(l, r):
            if r <= l + 1:
                cache[(l, r)] = 0
                return 0
            if (l, r) in cache:
                return cache[(l, r)]
            res = float('inf')
            for i in range(l, r):
                res = min(res, nums[i] + max(helper(l, i), helper(i + 1, r)))
            cache[(l, r)] = res
            # print(cache)
            return res

        nums = list(range(n + 1))
        cache = {}
        return helper(1, n + 1)


class Solution(object):
    def getMoneyAmount(self, n):
        """
        :type n: int
        :rtype: int
        """
        cache = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
        for lo in range(n - 1, 0, -1):
            for hi in range(lo + 1, n + 1):
                cache[lo][hi] = float('inf')
                for pivot in range(lo, hi):
                    cache[lo][hi] = min(
                        cache[lo][hi], pivot + max(cache[lo][pivot - 1],
                                                   cache[pivot + 1][hi]))
        return cache[1][n]