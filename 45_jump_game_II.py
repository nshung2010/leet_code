class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        #print(len(nums))
        res = 0
        start = 0
        temp_start = 0
        while start < len(nums)-1:
            #print('start', start)
            res+=1
            temp = 1
            upper_bound = min(nums[start], len(nums)-start-1)+1
            if start+nums[start] >= len(nums)-1:
                return res
            for i in range(1, upper_bound):
                if i+nums[start+i] >= temp:
                    temp_start = start+i
                    temp = i+nums[start+i] 
                #print('temp', temp)

            start = temp_start
            
        return res

sol = Solution()
a = sol.jump([1,2,3])
print(a)