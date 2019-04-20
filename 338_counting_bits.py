"""
Given a non negative integer number num. For every numbers i in the
range 0 ≤ i ≤ num calculate the number of 1's in their binary
representation and return them as an array.

Example 1:

Input: 2
Output: [0,1,1]
Example 2:

Input: 5
Output: [0,1,1,2,1,2]
"""
class Solution(object):
    def countBits(self, nums):
        """
        :type num: int
        :rtype: List[int]
        """
        if nums==0:
            return [0]
        dp=[0]*(nums+1)
        dp[1]=1
        for i in range(2,nums+1):
            q,r=divmod(i, 2)
            dp[i]=dp[q]+dp[r]
        return dp
