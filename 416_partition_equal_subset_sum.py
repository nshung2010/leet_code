"""
Given a non-empty array containing only positive integers, find if the
array can be partitioned into two subsets such that the sum of elements
in both subsets is equal.

Note:

Each of the array element will not exceed 100.
The array size will not exceed 200.


Example 1:

Input: [1, 5, 11, 5]

Output: true

Explanation: The array can be partitioned as [1, 5, 5] and [11].


Example 2:

Input: [1, 2, 3, 5]

Output: false

Explanation: The array cannot be partitioned into equal sum subsets.
"""
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        nums.sort(reverse=True)
        if sum(nums) % 2 != 0:
            return False
        target = sum(nums)//2
        n = len(nums)
        def dfs(i, target):
            # print(target, memo)
            if target in memo:
                return memo[target]
            memo[target] = False
            if target > 0:
                for j in range(i, n):
                    if dfs(j+1, target-nums[j]):
                        memo[target] = True
                        break
            return memo[target]
        memo = {0: True}
        return dfs(0, target)
