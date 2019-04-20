"""
Given a positive integer n, break it into the sum of at least two
positive integers and maximize the product of those integers.
Return the maximum product you can get.

Example 1:

Input: 2
Output: 1
Explanation: 2 = 1 + 1, 1 × 1 = 1.
Example 2:

Input: 10
Output: 36
Explanation: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36.
"""
class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        def dfs(num):
            if num==1:
                return 1
            if num in memo:
                return memo[num]
            max_product = 0
            for i in range(1, num):
                max_product = max(max_product, i*dfs(num-i), i*(num-i))
            memo[num] = max_product
            return max_product
        memo = {}

        return dfs(n)

class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp=[0]*(n+1)
        for i in range(2,n+1):
            res=float('-inf')
            for j in xrange(1,i):
                res=max(res,j*max(dp[i-j],i-j))
            dp[i]=res
        return dp[n]
