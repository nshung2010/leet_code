"""
A binary watch has 4 LEDs on the top which represent the hours (0-11),
and the 6 LEDs on the bottom represent the minutes (0-59).

Each LED represents a zero or one, with the least significant bit on
the right. Given a non-negative integer n which represents the number
of LEDs that are currently on, return all possible times the watch
could represent.

Example:

Input: n = 1
Return: ["1:00", "2:00", "4:00", "8:00", "0:01", "0:02", "0:04",
         "0:08", "0:16", "0:32"]
Note:
The order of output does not matter.
The hour must not contain a leading zero, for example "01:00" is not
valid, it should be "1:00".
The minute must be consist of two digits and may contain a leading zero,
for example "10:2" is not valid, it should be "10:02".
"""

class Solution(object):
    def readBinaryWatch(self, num):
        """
        :type num: int
        :rtype: List[str]
        """

        def get_total_combination_sum(nums, n, res):
            if n==0:
                return res
            results = set()
            for i, num in enumerate(nums):

                new_res = set([x + num for x in list(res)])
                results |= get_total_combination_sum(nums[:i]+nums[i+1:], n-1, new_res)
            return results
        def get_string_display(hour, minute):
            if minute == 0:
                return str(hour) + ':00'
            if minute <10:
                return str(hour) + ':0' + str(minute)
            return str(hour) + ':' + str(minute)
        if num == 0:
            return ['0:00']
        nums_hour = [1, 2, 4, 8]
        nums_minute = [1, 2, 4, 8, 16, 32]
        res = []
        for bit_hour in range(num+1):
            bit_minute = num - bit_hour
            hour_list = list(get_total_combination_sum(nums_hour, bit_hour, [0]))

            minute_list = list(get_total_combination_sum(nums_minute, bit_minute, [0]))

            for h in hour_list:
                for m in minute_list:
                    if 0<=h<12 and 0<=m<60:
                        res.append(get_string_display(h, m))

        return res

# Smart solution
def readBinaryWatch(self, num):
    return ['%d:%02d' % (h, m)
            for h in range(12) for m in range(60)
            if (bin(h) + bin(m)).count('1') == num]

# Better bactracking:
class Solution(object):
    def readBinaryWatch(self, num):
        """
        :type num: int
        :rtype: List[str]
        """
        def backtrack(hours, minutes, start):
            # print(hours, minutes, start)
            if len(hours)+len(minutes) > num:
                return
            if len(hours) + len(minutes) == num:
                h = sum(hours)
                m = sum(minutes)
                if h < 12 and m <60:
                    res.append('%d:%02d' % (h, m))
                    return
            for i in range(start, 10):
                if i<4:
                    backtrack(hours+[nums[i]], minutes, i+1)
                else:
                    backtrack(hours, minutes+[nums[i]], i+1)

        if num == 0:
            return ['0:00']
        res = []
        nums = [1, 2, 4, 8, 1, 2, 4, 8, 16, 32 ]
        backtrack([], [], 0)
        return res
