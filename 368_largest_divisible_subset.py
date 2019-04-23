"""
Given a set of distinct positive integers, find the largest subset such
that every pair (Si, Sj) of elements in this subset satisfies:

Si % Sj = 0 or Sj % Si = 0.

If there are multiple solutions, return any subset is fine.

Example 1:

Input: [1,2,3]
Output: [1,2] (of course, [1,3] will also be ok)
Example 2:

Input: [1,2,4,8]
Output: [1,2,4,8]
"""
class Solution(object):
    def largestDivisibleSubset(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums:
            return []
        nums=sorted(nums)
        dp=[[nums[i]] for i in range(len(nums))]
        max_len,res=1,dp[0]
        for i in range(1,len(nums)):
            for j in range(i):
                if nums[i]%nums[j]==0 and len(dp[i])<len(dp[j])+1:
                    dp[i]=dp[j]+[nums[i]]
                    if len(dp[i])>max_len:
                        max_len,res=len(dp[i]),dp[i]
        return res

# Another solution:

class Solution(object):
    def recur(path,lastIdx):
            for i in range(lastIdx+1,n):
                curr = nums[i]

                if curr % path[-1] == 0 :
                    if curr not in seen or seen[curr] < len(path):
                        seen[curr] = len(path)
                        recur(path + [curr],i)
            d[len(path)] = path

        n = len(nums)
        if n <= 1: return nums

        d = {};seen = {}
        nums.sort()
        for i in range(n-1):
            recur([nums[i]],i)
        maxKey = max(d.keys())
        return d[maxKey]
