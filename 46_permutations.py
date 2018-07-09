class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if not nums:
            return []
        if len(nums)==1:
            return [nums]
        res = []
        for i in range(len(nums)):
            temp = nums[0:i]+nums[i+1:]
            res+=([nums[i]]+v for v in self.permute(temp))
        return res
        
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if not nums:
            return []
        if len(nums)==1:
            return [nums]
        nums.sort()
        res = []
        for i in range(len(nums)):
            if i>0 and nums[i]==nums[i-1]:
                continue
            else:
                temp = nums[0:i]+nums[i+1:]
                res+=([nums[i]]+v for v in self.permuteUnique(temp))
        return res
        