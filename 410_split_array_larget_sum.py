"""
Given an array which consists of non-negative integers and an integer m,
you can split the array into m non-empty continuous subarrays. Write an
algorithm to minimize the largest sum among these m subarrays.

Note:
If n is the length of array, assume the following constraints are satisfied:

1 ≤ n ≤ 1000
1 ≤ m ≤ min(50, n)
Examples:

Input:
nums = [7,2,5,10,8]
m = 2

Output:
18

Explanation:
There are four ways to split nums into two subarrays.
The best way is to split it into [7,2,5] and [10,8],
where the largest sum among the two subarrays is only 18.
"""
class Solution(object):
    def splitArray(self, nums, m):
        """
        :type nums: List[int]
        :type m: int
        :rtype: int
        """
        if m==1:
            return sum(nums)
        sums = [0]
        current_sum = 0
        for i, num in enumerate(nums):
            current_sum += num
            sums.append(current_sum)
        dp = [[0] * len(nums) for _ in range(m)]
        for i in range(len(nums)):
            dp[0][i] = sums[i+1]
        for row in range(1, m):
            for col in range(row, len(nums)):
                res = float('inf')
                for index in range(row-1, col):
                    res = min(res, max(dp[row-1][index], sums[col+1]-sums[index+1]))
                dp[row][col] = res
        # print(dp)
        return dp[-1][-1]

# Binary search
# -*- coding: utf-8 -*-
class Solution(object):
    def splitArray(self, nums, m):
        """
        :type nums: List[int]
        :type m: int
        :rtype: int
        """
        L, R = max(nums), sum(nums) + 1

        ans = 0
        while L < R:
            mid = (L + R) / 2
            if self.guess(mid, nums, m):
                ans = mid
                R = mid
            else:
                L = mid + 1
        return ans

    # @staticmethod
    def guess(self, mid, nums, m):
        sum = 0
        for i in range(0, len(nums)):
            if sum + nums[i] > mid:
                m -= 1
                sum = nums[i]
            else:
                sum += nums[i]
        return m >= 1
