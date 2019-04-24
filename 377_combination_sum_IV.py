"""
Given an integer array with all positive numbers and no duplicates,
find the number of possible combinations that add up to a positive
integer target.

Example:

nums = [1, 2, 3]
target = 4

The possible combination ways are:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)

Note that different sequences are counted as different combinations.

Therefore the output is 7.
"""
class Solution(object):
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        def helper(target):

            count = 0
            for num in nums:
                if num > target:
                    return count
                elif num == target:
                    count += 1
                else:
                    new_target = target - num
                    if new_target in cache:
                        count += cache[new_target]
                    else:
                        temp = helper(new_target)
                        count += temp
                        cache[new_target] = temp
            return count
        cache = {}
        nums.sort()
        return helper(target)

# Better solution
class Solution(object):
    def combinationSum4(self, nums, target):
        nums.sort()
        dp=[0]*(target+1)
        dp[0]=1 # if num == target
        for i in xrange(1,target+1):
            for num in nums:
                if num>i:
                    break
                dp[i]+=dp[i-num]
        return dp[target]

class Solution:
    def combinationSum4(self, nums, target):
        nums.sort()
        # memory(a dict) to remember whether we have dfs(m), if we did, remember the number of combs of m
        mem = {}
        # dfs return the number of combinations in which every comb add up to rest
        def dfs(rest):
            ans = 0
            for i in range(0, len(nums)):
                if nums[i] > rest:
                    return ans
                if nums[i] == rest:
                    ans += 1
                elif nums[i] < rest:
                    # if we have dfs(rest - nums[i]) before, no need to do it again
                    if rest - nums[i] in mem:
                        ans += mem[rest - nums[i]]
                    else:
                        # tmp is the number of combinations in which every comb add up to 'rest - nums[i]'
                        tmp = dfs(rest - nums[i])
                        mem[rest - nums[i]] = tmp
                        ans += tmp
            return ans
        return dfs(target)
