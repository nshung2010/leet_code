# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

#prob: 617
class Solution(object):
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        if not t1 and not t2:
            return None
        elif not t2:
            return t1
        elif not t1:
            return t2
        else:
            a = TreeNode(t1.val + t2.val)
            a.left = self.mergeTrees(t1.left, t2.left)
            a.right = self.mergeTrees(t1.right, t2.right)
        return a

# prob: 94
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        t, stack = [], [root]
        #return stack.pop()
        while stack:
            x = stack.pop()
            if isinstance(x,TreeNode):
                stack.append(x.left)
                stack.append(x.val)
                stack.append(x.right)
            elif x is not None:
                t.append(x)
            #print(t)
        return t[::-1]

#prob: 145
class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        traversal, stack = [], [root]
        #x = stack.pop()
        #return x.left
                #return stack.pop()
        while stack:
            x = stack.pop()
            if x:
                traversal.append(x.val)
                stack.append(x.left)
                stack.append(x.right)
        return traversal[::-1]

#prob: 98
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):

    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        self.all = list()
        self.get_all_val(root)
        return self.all == sorted(self.all) and len(set(self.all)) == len(self.all)
    def get_all_val(self, root):
        """
        :type root: TreeNode
        :rtype: self.all which is a matrix contain all values in TreeNode
        """
        if root is None:
            return
        self.get_all_val(root.left)
        self.all.append(root.val)
        self.get_all_val(root.right)


#501:
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from collections import defaultdict
class Solution(object):
    def findMode(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root is None:
            return list()
        freq = defaultdict(int)
        self.count_node(root, freq)
        max_freq = max(freq.values())
        #return freq.values()
        x = [k for k, v in freq.items() if v == max_freq]
        return x
    def count_node(self, root, freq):
        """
        :type root: TreeNode
        :rtype: self.all which is a matrix contain all values in TreeNode
        """
        if root is None:
            return
        freq[root.val] += 1
        self.count_node(root.left, freq)
        self.count_node(root.right, freq)
        return

#96:
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        x = [0]*(n+1)
        x[0] = 1
        x[1] = 1
        for i in range(2,n+1,1):
            for j in range(i):
                x[i] += x[j] * x[i-1-j]
        return x[n]

#95:
#628:
#461:
class Solution(object):
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        n_x = x.bit_length()
        n_y = y.bit_length()
        n = max(n_x, n_y)
        x_b = bin(x)[2:].zfill(n)
        y_b = bin(y)[2:].zfill(n)
        hammingDistance = 0
        for i in range(n):
            if x_b[i] <> y_b[i]:
                hammingDistance += 1
        return hammingDistance

#561
class Solution(object):
    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums_sort = sorted(nums)
        print(nums_sort)

        a = nums_sort[::2]
        return sum(a)

class Solution(object):
    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        return sum(sorted(nums)[::2])

#476   number complement
class Solution(object):
    def findComplement(self, num):
        """
        :type num: int
        :rtype: int
        """
        n = num.bit_length()
        return 2**(n)-1-num

#500 Keyboard Row
class Solution(object):
    def findWords(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        row1 = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']
        row2 = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l']
        row3 = ['z', 'x', 'c', 'v', 'b', 'n', 'm']

        for each w in words:

def findWords(self, words):
    return filter(re.compile('(?i)([qwertyuiop]*|[asdfghjkl]*|[zxcvbnm]*)$').match, words)

class Solution:
    def findWordsRe(self, words):
        return list(filter(re.compile('(?i)([qwertyuiop]*|[asdfghjkl]*|[zxcvbnm]*)$').match, words))

    def findWordsRe2(self, words):
        return [word for word in words if re.compile('(?i)([qwertyuiop]*|[asdfghjkl]*|[zxcvbnm]*)$').match(word)]

    def findWordsSet(self, words):
        row1 = set('qwertyuiopQWERTYUIOP')
        row2 = set('asdfghjklASDFGHJKL')
        row3 = set('zxcvbnmZXCVBNM')
        return list(filter(lambda x: set(x).issubset(row1) or set(x).issubset(row2) or set(x).issubset(row3), words))

    def findWordsSet2(self, words):
        row1 = set('qwertyuiopQWERTYUIOP')
        row2 = set('asdfghjklASDFGHJKL')
        row3 = set('zxcvbnmZXCVBNM')
        return [word for word in words if set(word).issubset(row1) or set(word).issubset(row2) or set(word).issubset(row3)]

import timeit, re, random, string

words = [''.join(random.choices(string.ascii_lowercase, k=5)) for i in range(1000)]
print(timeit.timeit(lambda: Solution().findWordsRe(words), number=1000), "(re with filter)")
print(timeit.timeit(lambda: Solution().findWordsRe2(words), number=1000), "(re with list comprehension)")
print(timeit.timeit(lambda: Solution().findWordsSet(words), number=1000), "(set with filter)")
print(timeit.timeit(lambda: Solution().findWordsSet2(words), number=1000), "(set with list comprehension)")

#557 Reverse words in a string III
def reverseWords(self, s):
    return ' '.join(s.split()[::-1])[::-1]
    return ' '.join(x[::-1] for x in s.split())
#575 Distribute candies
class Solution(object):
    def distributeCandies(self, candies):
        """
        :type candies: List[int]
        :rtype: int
        """
        candyType = set(candies)
        return min(len(candyType), len(candies)/2)

#1: two sum:
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        nums_sort = sorted(nums)
        ind = sorted(range(len(nums)),key=lambda x:nums[x])
        n = len(nums_sort)
        i = 0
        j = n-1
        while i<j:
            if nums_sort[i] + nums_sort[j] > target:
                j -= 1
            elif nums_sort[i] + nums_sort[j] < target:
                i += 1
            elif nums_sort[i] + nums_sort[j] == target:
                x = [ind[i], ind[j]]
                i = j+1
        return x

    class Solution(object):
    def twoSum(self, nums, target):
        if len(nums) <= 1:
            return False
        buff_dict = {}
        for i in range(len(nums)):
            if nums[i] in buff_dict:
                return [buff_dict[nums[i]], i]
            else:
                buff_dict[target - nums[i]] = i


#2: Add two numbers
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """

        ret = ListNode(0)
        cur = ret
        add = 0

        while l1 or l2 or add:
            val = (l1.val if l1 else 0) + (l2.val if l2 else 0) + add
            add = val / 10
            cur.next = ListNode(val % 10)
            cur = cur.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        return ret.next

    class Solution:
    def addTwoNumbers(self, l1, l2):
        addends = l1, l2
        dummy = end = ListNode(0)
        carry = 0
        while addends or carry:
            carry += sum(a.val for a in addends)
            addends = [a.next for a in addends if a.next]
            end.next = end = ListNode(carry % 10)
            carry /= 10

    class Solution:
    def addTwoNumbers(self, l1, l2):
        def toint(node):
            return node.val + 10 * toint(node.next) if node else 0
        def tolist(n):
            node = ListNode(n % 10)
            if n > 9:
                node.next = tolist(n / 10)
            return node
        return tolist(toint(l1) + toint(l2))

    class Solution:
    def addTwoNumbers(self, l1, l2):
        def toint(node):
            return node.val + 10 * toint(node.next) if node else 0
        n = toint(l1) + toint(l2)
        first = last = ListNode(n % 10)
        while n > 9:
            n /= 10
            last.next = last = ListNode(n % 10)
        return first

#4: Median of two sorted arrays

#21: Merge Two Sorted Lists:
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        l=ListNode(None)
        if not l1 and not l2:
            return None
        elif not l2:
            return l1
        elif not l1:
            return l2
        else:
            if l1.val <=l2.val:
                l.val = l1.val
                l.next = self.mergeTwoLists(l1.next, l2)
            elif l1.val > l2.val:
                l.val = l2.val
                l.next = self.mergeTwoLists(l1, l2.next)
        return l

#3: Longest substring without repeating characters

class Solution(object):

    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        i=0
        max_length = 0

        while i<len(s):
            j=i+1+max_length
                while j<len(s)+1
                    t = s[i:j]
                    if j < len(s) and s[j] in t:
                        j = len(s)
                        i = i+t.index(s[j])

                    j = j+1

                max_length=max(max_length,len(t))

        return max_length

class Solution:
    # @return an integer
    def lengthOfLongestSubstring(self, s):
        start = maxLength = 0
        usedChar = {}

        for i in range(len(s)):
            if s[i] in usedChar and start <= usedChar[s[i]]:
                start = usedChar[s[i]] + 1
            else:
                maxLength = max(maxLength, i - start + 1)

            usedChar[s[i]] = i

        return maxLength

#4 Median of two sorted arrays
import math
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        l = len(nums1) + len(nums2)

        if l%2==1:
            return self.findKth(nums1,nums2,l//2)
        else:
            return (self.findKth(nums1,nums2,l//2-1)+self.findKth(nums1,nums2,l//2))/2.0

    def findKth(self,nums1, nums2, k):
        if len(nums1)>len(nums2):
            nums1, nums2 = nums2, nums1
        if not nums1:
            return nums2[k]
        if k==len(nums1)+len(nums2)-1:
            return max(nums1[-1], nums2[-1])

        i = min(len(nums1)-1,k//2)
        j = min(len(nums2)-1,k-i)
        if nums1[i] > nums2[j]:
            return self.findKth(nums1[:i],nums2[j:],i)
        else:
            return self.findKth(nums1[i:], nums2[:j],j)

#5 longest palaindromic substring


class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        n = ""
        for i in range(len(s)):
            n1 = self.expand_around_center(s,i,i)
            n2 = self.expand_around_center(s,i,i+1)
            if max(len(n1),len(n2))>len(n):
                if len(n1)>len(n2):
                    n=n1
                else:
                    n = n2
        return n

    def expand_around_center(self, s, l, r):
        while l>=0 and r <len(s) and s[l]==s[r]:
            l=l-1
            r=r+1
        return s[l+1:r]

#6 ZigZag conversion
class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        n = len(s)
        if numRows ==1 or numRows > n-1:
            return s
        step = 2*numRows-2
        x = [""]*numRows
        for irow in range(numRows):
            i=irow

            while i<n:
                if irow==0 or irow==numRows-1:
                    x[irow]+=s[i]
                    i=i+2*numRows-2
                else:
                    x[irow]+=s[i]
                    if i+step<n:
                        x[irow]+=s[i+step]
                    i=i+2*numRows-2
            step = step-2
        return "".join(x)

#7 ReVerse interger
class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x >= 0:
            s = str(x)
            s = s[::-1]
            return int(s) if int(s)<(1<<31)-1 else 0
        else:
            s = str(x*(-1))
            s = s[::-1]
            return -1*int(s) if int(s)<(1<<31)-1 else 0
#8: Aoi - string to integer
class Solution(object):
   class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        str = str.lstrip()
        sign = 1
        if len(str)==0:
            return 0

        if str[0] == "+":
            str=str[1:]
        elif str[0]=="-":
            sign = -1
            str=str[1:]
        elif not str[0].isdigit():
            return 0

        if len(str)==0:
            return 0

        s = str[0]
        i = 0
        x = 0
        while i < len(str) and str[i].isdigit():
            s = int(str[i])
            x = s+x*10
            i = i+1

        return max(-2**31, min(sign * x,2**31-1))

class Solution(object):
    def myAtoi(self, s):
        """
        :type str: str
        :rtype: int
        """
        ###better to do strip before sanity check (although 8ms slower):
        #ls = list(s.strip())
        #if len(ls) == 0 : return 0
        if len(s) == 0 : return 0
        ls = list(s.strip())

        sign = -1 if ls[0] == '-' else 1
        if ls[0] in ['-','+'] : del ls[0]
        ret, i = 0, 0
        while i < len(ls) and ls[i].isdigit() :
            ret = ret*10 + ord(ls[i]) - ord('0')
            i += 1
        return max(-2**31, min(sign * ret,2**31-1))

#9 Palindrome Number (sysmetric)
class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x<0:
            return False
        s = str(x)
        y = int(s[::-1])
        return True if y==x else False
#11 Container with most water
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        max_area = 0
        l = 0
        r = len(height)-1

        while l<r:
            max_area=max(max_area,min(height[l],height[r])*(r-l))
            if height[l] < height[r]:
                l+=1
            else:
                r-=1
        return max_area
#12 Integer to Roman
class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        if num < 1 or num > 3999:
            return None
        values = [ 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 ]
        numerals = [ "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" ]
        res=""
        for i, v in zip(values, numerals):
            res+=(num//i)*v
            num%=i
        return res

#13 Roman to Integer
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        Roman = {'M':1000, 'D':500, 'C':100, 'L':50, 'X':10, 'V':5, 'I':1}
        sum = 0
        for i in range(len(s)):
            if i<len(s)-1 and Roman[s[i]]<Roman[s[i+1]]:
                sum = sum - Roman[s[i]]
            else:
                sum = sum + Roman[s[i]]
        return sum
#14 Longest common prefix

class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs or any (x =="" for x in strs):
            return ""
        prefix = strs[0]
        for i in range(len(prefix)):
            s = prefix[i]
            for j in range(len(strs)):
                s_compare = strs[j]
                if len(s_compare)<i+1 or s!=s_compare[i]:
                    return pref

class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs or any (x =="" for x in strs):
            return ""
        prefix = strs[0]
        l=0
        r=len(prefix)-1
        while l<=r:
            m = (l+r)//2
            if self.isCommonPrefix(strs,m):
                l = m+1
            else:
                r = m-1

        return prefix[:(l+r)//2+1]

    def isCommonPrefix(self, strs,m):
        s = strs[0][0:m+1]
        if all(x.startswith(s) for x in strs):
            return True
        return False

# 15 3sum
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        if len(nums)<3:
            return []
        if len(nums)==3:
            if sum(nums)==0:
                return [nums]
            else:
                return []

        x = self.threeSum(nums[0:len(nums)-1])
        y = self.twoSum(nums[0:len(nums)-1],nums[-1])
        if y:
            res = x+y
        else:
            res =  x
        return res

    def twoSum(self,nums,x):
        l = 0
        r = len(nums)-1
        res = []
        while l<r:
            if nums[l]+nums[r]+x>0:
                res.append([nums[l],nums[r]])
                r-=1
            elif nums[l]+nums[r]+x<0:
                l+=1
            else:
                res.append([nums[l],nums[r]])
                l+=1
        if res:
            return [y+[x] for y in res]
        else:
            return []

class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) < 3:
            return []
        nums.sort()
        res = set()
        for i, v in enumerate(nums[:-2]):
            if i >= 1 and v == nums[i-1]:
                continue
            d = {}
            for x in nums[i+1:]:
                if x not in d:
                    d[-v-x] = 1
                else:
                    res.add((v, -v-x, x))
        return map(list, res)

class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) <3:
            return []
        nums.sort()
        res = set()
        for i,x in enumerate(nums[:-2]):
            if i>=1 and nums[i]==nums[i-1]:
                continue
            d={}
            for v in nums[i+1:]:
                if v not in d:
                    d[-v-x]=1
                else:
                    res.add((v,x,-v-x))
        return map(list,res)

#16  3sum closest
class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if len(nums)<3:
            return 0
        nums.sort()
        hold_diff = sys.maxint

        for i, x1 in enumerate(nums[:-2]):
            if i>=1 and x1 == nums[i-1]:
                continue
            l,r = i+1, len(nums)-1
            while l<r:
                total = x1+nums[l]+nums[r]
                diff = abs(total-target)
                if diff == 0:
                    return total

                if diff< hold_diff:
                    res = total
                    hold_diff = diff
                if total < target:
                    l+=1
                else:
                    r-=1

        return res

#17: Letter combination of a phone number
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        map = {'2':'abc','3':'def', '4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
        if len(digits)==0:
            return []

        if len(digits)==1:
            return [x for x in list(map[digits])]

        comb = self.letterCombinations(digits[:-1])
        last = [x for x in list(map[digits[-1]])]
        return [x+y for x in comb for y in last]
#18: kSum
class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        nums.sort()
        results = []
        self.findKsum(nums, target, 4, [], results)
        return results

    def findKsum(self, nums, target, N, result, results):
        if len(nums) < N or N<2 or target < nums[0]*N or target>nums[-1]*N :
            return []
        if N ==2:
            l=0
            r = len(nums)-1
            while l<r:
                s = nums[l]+nums[r]
                if s == target:
                    results.append(result + [nums[l], nums[r]])
                    l +=1
                    while l<r and nums[l]==nums[l-1]:
                        l += 1
                elif s<target:
                    l+=1
                else:
                    r-=1
        else:
            for i, x in enumerate(nums[:-(N-1)]):
                if i>0 and nums[i]==nums[i-1]:
                    continue
                self.findKsum(nums[i+1:], target - x, N-1, result + [x], results)
        return results

#20 Valid Parentheses
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        brackets = {"]":"[", "}":"{", ")":"("}
        for char in s:
            if char in brackets.values():
                stack.append(char)
            elif char in brackets.keys():
                if stack ==[] or brackets[char] != stack.pop():
                    return False
            else:
                return False
        return stack ==[]

#19:Remove Nth Node From End of List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        self.next = head
        first = self
        second = self

        for i in range(n+1):
            first = first.next

        while first:
            first = first.next
            second = second.next

        second.next = second.next.next
        return self.next

#22: Generate Parenthesis
class Solution(object):
    def generateParenthesis(self, N):
        ans = []
        def backtrack(S = '', left = 0, right = 0):
            print(S)
            if len(S) == 2 * N:
                ans.append(S)
                return
            if left < N:
                backtrack(S+'(', left+1, right)
            if right < left:
                backtrack(S+')', left, right+1)

        backtrack()
        return ans
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        i=0
        n = len(nums)
        while i<n-1:
            if nums[i]==nums[i+1]:
                nums.pop(i)
                n-=1
                i-=1
            i+=1
        return n
#27 Remove element
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        n=0
        for i in range(len(nums)):
            if nums[i]!=val:
                nums[n]=nums[i]
                n+=1

        return n
#28: implement strSTR
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        return haystack.find(needle)
#10 Regular expression matching
class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        if not p:
            return not s

        first_match = bool(s) and p[0] in {s[0], '.'}
        if len(p)>=2 and p[1] == '*':
            return (self.isMatch(s,p[2:]) or
                    first_match and self.isMatch(s[1:],p))
        else:
            return first_match and self.isMatch(s[1:],p[1:])

class Solution(object):
    def isMatch(self, text, pattern):
        memo = {}
        def dp(i, j):
            if (i, j) not in memo:
                if j == len(pattern):
                    ans = i == len(text)
                else:
                    first_match = i < len(text) and pattern[j] in {text[i], '.'}
                    if j+1 < len(pattern) and pattern[j+1] == '*':
                        ans = dp(i, j+2) or first_match and dp(i+1, j)
                    else:
                        ans = first_match and dp(i+1, j+1)

                memo[i, j] = ans
            return memo[i, j]

        return dp(0, 0)
#23 Merge k sorted list
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        k = len(lists)
        interval = 1
        while interval < k:
            for i in range(0,k-interval,interval*2):
                lists[i] = self.merge2Lists(lists[i],lists[i+interval])
            interval*=2
        return lists[0] if k>0 else lists
    def merge2Lists(self, l1, l2):

        point = head = ListNode(0)

        while l1 and l2:
            if l1.val<= l2.val:
                point.next = l1
                l1= l1.next

            else:
                point.next = l2
                l2= l1
                l1 = point.next.next

            point = point.next

        if not l1:
            point.next = l2
        else:
            point.next = l1
        return head.next

#24 Swap nodes in pairs
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        pre, pre.next = self, head
        while pre.next and pre.next.next:
            a = pre.next
            b = a.next
            pre.next,b.next, a.next = b, a, b.next
            pre = a
        return self.next

def swapPairs(self, head):
    dummy = cur = ListNode(-1)
    dummy.next = head

    while cur.next and cur.next.next:
        p1, p2 = cur.next, cur.next.next
        cur.next, p1.next, p2.next = p2, p2.next, p1
        cur = cur.next.next
    return dummy.next

class Solution(object):
    def swapPairs(self, head):
        if head and head.next:
            tmp = head.next
            head.next = self.swapPairs(tmp.next)
            tmp.next = head
            return tmp
        return head

#31: Next Permutation:
class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        r=len(nums)-1
        while r-1 >=0 and nums[r]<=nums[r-1]:
            r-=1
        #print(r)
        if r==0:
            nums.sort()
        else:
            k = nums[r-1]
            i=r
            while i<len(nums) and nums[i]>k:
                i+=1
            #print(i)
            nums[r-1],nums[i-1] = nums[i-1], nums[r-1]
            nums[r:]=reversed(nums[r:])

#32 Longest valid parentheses:
class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s)<2:
            return 0
        dp=[0]*len(s)
        max_len = 0
        for i in range(1,len(s)):
            if s[i]==")" and s[i-1]=="(":
                if i<2:
                    dp[i]=2
                else:
                    dp[i]=dp[i-2]+2
            elif s[i]==")" and s[i-1]==")":
                if s[i-dp[i-1]-1]=="(" and i-dp[i-1]-1>=0:
                    if i-dp[i-1]-2>=0:
                        dp[i]=dp[i-1]+dp[i-dp[i-1]-2]+2
                    else:
                        dp[i]=dp[i-1]+2

            max_len=max(max_len,dp[i])
        return max_len

class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        max_len=0
        l=r=0
        for x in s:
            if x=="(":
                l+=1
            else:
                r+=1
            if r>l:
                l=r=0
            if l==r:
                max_len=max(max_len,2*l)

        l=r=0
        for x in reversed(s):
            if x=="(":
                l+=1
            else:
                r+=1
            if l>r:
                l=r=0
            if l==r:
                max_len=max(max_len,2*l)
        return max_len

#33 search in rotaged sorted array:
class Solution:
    # @param {integer[]} numss
    # @param {integer} target
    # @return {integer}
    def search(self, nums, target):
        if not nums:
            return -1

        low, high = 0, len(nums) - 1

        while low <= high:
            mid = (low + high) / 2
            if target == nums[mid]:
                return mid

            if nums[low] <= nums[mid]:
                if nums[low] <= target <= nums[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                if nums[mid] <= target <= nums[high]:
                    low = mid + 1
                else:
                    high = mid - 1

        return -1
#34:
#35: search insert position
class Solution(object):
def searchInsert(self, nums, key):
    if key > nums[len(nums) - 1]:
        return len(nums)

    if key < nums[0]:
        return 0

    l, r = 0, len(nums) - 1
    while l <= r:
        m = (l + r)/2
        if nums[m] > key:
            r = m - 1
            if r >= 0:
                if nums[r] < key:
                    return r + 1
            else:
                return 0

        elif nums[m] < key:
            l = m + 1
            if l < len(nums):
                if nums[l] > key:
                    return l
            else:
                return len(nums)
        else:
            return m

#36 Valid Sudoku
def isValidSudoku(self, board):
    return (self.is_row_valid(board) and
            self.is_col_valid(board) and
            self.is_square_valid(board))

def is_row_valid(self, board):
    for row in board:
        if not self.is_unit_valid(row):
            return False
    return True

def is_col_valid(self, board):
    for col in zip(*board):
        if not self.is_unit_valid(col):
            return False
    return True

def is_square_valid(self, board):
    for i in (0, 3, 6):
        for j in (0, 3, 6):
            square = [board[x][y] for x in range(i, i + 3) for y in range(j, j + 3)]
            if not self.is_unit_valid(square):
                return False
    return True

def is_unit_valid(self, unit):
    unit = [i for i in unit if i != '.']
    return len(set(unit)) == len(unit)

#sol 2:
seen=[x for i, row in enumerate(board)
            for j, c in enumerate(row)
                if c!='.'
                    for x in ((c,i),(j,c),(i/3,j/3,c))]
return len(seen)==len(set(seen))

#38 count and say:
class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        result="1"
        for _ in range(n-1):
            pre = result
            result = ""
            j=0
            while j<len(pre):
                cur = pre[j]
                count=1
                j+=1
                while j<len(pre) and pre[j]==pre[j-1]:
                    count+=1
                    j+=1
                result = result + str(count) + str(cur)
        return result

# combination sum
class Solution(object):
    def combinationSum(self, candidates, target):
        res = []
        candidates.sort()
        self.dfs(candidates, target, 0, [], res)
        return res

    def dfs(self, nums, target, index, path, res):
        if target < 0:
            return  # backtracking
        if target == 0:
            res.append(path)
            return
        for i in
        range(index, len(nums)):
            if nums[i]>target:
                break
            self.dfs(nums, target-nums[i], i, path+[nums[i]], res)

#37 SUDOKU solver:
class Solution(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        self.board = board
        self.solve()


    def findUnAssign(self):
        for row in range(9):
            for col in range(9):
                if self.board[row][col]==".":
                    return row, col
        return -1, -1
    def isSafe(self, row, col, ch):
        block_row = row - row%3
        block_col = col - col%3
        return self.isSafeRow(row, ch) and self.isSafeCol(col, ch) and self.isSafeBlock(block_row, block_col, ch)

    def isSafeRow(self, row, ch):
        for i in range(9):
            if self.board[row][i]==ch:
                return False
        return True

    def isSafeCol(self, col, ch):
        for i in range(9):
            if self.board[i][col]==ch:
                return False
        return True
    def isSafeBlock(self, block_row, block_col, ch):
        for i in range(block_row, block_row+3):
            for j in range(block_col, block_col+3):
                if self.board[i][j]==ch:
                    return False
        return True
    def solve(self):
        i,j = self.findUnAssign()
        if i==-1 and j==-1:
            return True
        for ch in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            if self.isSafe(i,j, ch):
                self.board[i][j]=ch
                if self.solve():
                    return True
            self.board[i][j]="."
        return False

#SUDOKU solver:
class Solution(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        self.board = board
        self.val = self.possibleVal()
        self.solve()

    def possibleVal(self):
        all_val = '123456789'
        val, d ={},{}
        for i in range(9):
            for j in range(9):
                ele = self.board[i][j]
                if ele !=".":
                    d[("r",i)]=d.get(("r",i),[]) + [ele]
                    d[("c",j)]=d.get(("c",j),[]) + [ele]
                    d[(i//3,j//3)]=d.get((i//3,j//3),[]) + [ele]
                else:
                    val[(i,j)] = []

        for (i,j) in val.keys():
            bad_val = d.get(("r",i),[]) + d.get(("c",j),[]) + d.get((i//3,j//3),[])
            val[(i,j)] = [n for n in all_val if n not in bad_val]
        return val

    def solve(self):
        if len(self.val) == 0:
            return True
        k = min(self.val.keys(), key=lambda x:len(self.val[x]))
        for num in self.val[k]:
            update={k:self.val[k]}
            if self.isValid(num, k, update):
                if self.solve():
                    return True
            self.undo(k , update)
        return False

    def isValid(self, num, k, update):
        i, j = k
        self.board[i][j] = num
        del self.val[k]
        for ind in self.val.keys():
             if num in self.val[ind]:
                    if ind[0]==i or ind[1]==j or (ind[0]//3,ind[1]//3)==(i//3,j//3):
                        update[ind] = num
                        self.val[ind].remove(num)
                        if len(self.val[ind])==0:
                            return False
        return True

    def undo(self, k , update):
        self.board[k[0]][k[1]] = "."

        for x in update.keys():
            if x not in self.val:
                self.val[x] = update[x]
            else:
                self.val[x].append(update[x])


#40 combination sum II:

class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        result = []
        temp = []
        candidates.sort(reverse=True)

        self.util(candidates, target, result, temp)
        return result

    def util(self, nums, target, result, temp):

        for i in range(len(nums)):
            if nums[i] == target and (temp + [nums[i]] not in result):
                result.append(temp + [nums[i]])
            elif nums[i] < target:
                self.util(nums[i + 1:], target - nums[i], result, temp + [nums[i]])

        return

class Solution:
    def combinationSum2(self, candidates, target):

        def dfs(i, val, path):
            while i < len(candidates):
                num = candidates[i]
                val_ = val + num
                path_ = path + [num]
                if val_ > target:
                    return
                elif val_ == target:
                    ans.append(path_)
                    return
                dfs(i+1, val_, path_)
                while i<len(candidates)-1 and candidates[i]==candidates[i+1]:
                    i += 1
                i += 1

        candidates = sorted(candidates)
        ans = []
        dfs(0, 0, [])
        return ans

class Solution(object):
    def combinationSum2(self, candidates, target):
        res = []
        candidates.sort()
        self.dfs(candidates, target, 0, [], res)
        return res

    def dfs(self, nums, target, index, path, res):
        if target < 0:
            return  # backtracking
        if target == 0:
            res.append(path)
            return
        for i in xrange(index, len(nums)):
            if i>0 and nums[i]==nums[i-1]:
                continue
            if nums[i]>target:
                break
            self.dfs(nums[0:i]+nums[i+1:], target-nums[i], i, path+[nums[i]], res)


#53: maximum subarray
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        for i in range(1, len(nums)):
            nums[i] = max(nums[i], nums[i]+nums[i-1])
        return max(nums)

class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return float('-inf')
        if len(nums) ==1:
            return nums[0]
        left = 0
        right = len(nums)
        mid = (right+left)//2
        max_left = self.maxSubArray(nums[left:mid])
        max_right = self.maxSubArray(nums[mid:right])
        max_cross = self.cross_max(nums)
        #print(max_left)
        #print(max_right)
        #print(max_cross)
        return max(max_left, max_right, max_cross)

    def cross_max(self, nums):
        if len(nums) == 1:
            return nums[0]
        left = 0
        right = len(nums)
        mid = (right+left)//2
        max_left = 0
        max_right = 0
        sum_temp = 0
        for i in range(mid,left-1,-1):
            sum_temp+=nums[i]
            max_left = max(max_left, sum_temp)
        #rint(max_left)
        sum_temp = 0
        for i in range(mid+1, right):
            sum_temp+=nums[i]
            max_right = max(max_right, sum_temp)
        #rint(max_right)
        if max_left == 0 and max_right ==0:
            return nums[mid]
        return max_left + max_right


#42 trapping rain water:
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if not height:
            return 0

        i_max = height.index(max(height))

        i = 0
        sum_area = 0
        while i < i_max:
            j=i+1
            while j<len(height)-1 and height[j]<height[i]:
                j+=1
            sum_area +=height[i]*(j-i)
            i=j

        i = len(height)-1
        while i>i_max:
            j=i-1
            while j>=0 and height[j]<height[i]:
                j-=1
            sum_area +=height[i]*(i-j)
            i=j

        return sum_area+max(height)-sum(height)


#43 multiply string:
class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        d = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}

        n1 = 0
        n2 = 0

        l1 = len(num1)-1
        l2 = len(num2)-1

        for i in range(len(num1)):
            num = num1[i]
            n1 += d[num]*(10**(l1))
            l1-=1

        for j in range(len(num2)):
            num = num2[j]
            n2 += d[num]*(10**(l2))
            l2-=1

        return str(n1*n2)
def multiply(num1, num2):
    product = [0] * (len(num1) + len(num2))
    pos = len(product)-1

    for n1 in reversed(num1):
        tempPos = pos
        for n2 in reversed(num2):
            product[tempPos] += int(n1) * int(n2)
            product[tempPos-1] += product[tempPos]/10
            product[tempPos] %= 10
            tempPos -= 1
        pos -= 1

    pt = 0
    while pt < len(product)-1 and product[pt] == 0:
        pt += 1

    return ''.join(map(str, product[pt:]))


#44: Wildcard matching
"""
class Solution(object):
    def isMatch(self, s, p):

        :type s: str
        :type p: str
        :rtype: bool

        m, n = len(s), len(p)
        dp=[[False]*(n+1) for _ in range(m+1)]
        dp[0][0] = True
        for j in range(n):
            if p[j] == '*':
                dp[0][j+1] = dp[0][j]
        for i in range(m):
            for j in range(n):
                if p[j] == '*':
                    dp[i+1][j+1] = dp[i+1][j] or dp[i][j] or dp[i][j+1]
                else:
                    last_match = p[j] in {s[i], "?"}
                    dp[i+1][j+1]= last_match and dp[i][j]
        return dp[m][n]

A = Solution()
c = A.isMatch("adceb", "*a*b")
print(c)
"""

class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        transfer = {}
        state = 0

        for char in p:
            if char == '*':
                transfer[state, char] = state
            else:
                transfer[state, char] = state + 1
                state += 1

        accept = state
        print(accept)
        print(transfer)
        state = set([0])
        print(state)
        for char in s:
            state = set([transfer.get((at, token)) for at in state for token in [char, '*', '?']])
            print(state)
        print(state)
        return accept in state

A = Solution()
c = A.isMatch("adceb", "a*b*")
print(c)

"""
# fast solution
class Solution(object):
    def isMatch(self, s, p):

        #:type s: str
        #:type p: str
        #:rtype: bool

        # b is the expression
        def xmatch(a,b):
            if(len(a)!=len(b)):
                return False
            for i in xrange(len(a)):
                if((not b[i]=='?') and (a[i]!=b[i])):
                    return False
            return True
        if(p.find("*")==-1):
            return xmatch(s,p)
        #preprocessing
        tail=0
        for i in xrange(len(p)-1,-1,-1):
            if(p[i]=='*'):
                break
            else:
                tail+=1
        head=p.find("*")
        if(head!=0):
            if(not xmatch(s[:head],p[:head])):
                return False
        if(tail!=0):
            if(not xmatch(s[-tail:],p[-tail:])):
                return False
        if(len(s)<head+tail):
            return False
        s=s[head:len(s)-tail]
        p=p[head:len(p)-tail].strip("*")
        dp=p.split("*")
        while("" in dp):
            dp.remove("")
        start=0
        for item in dp:
            if(start>=len(s)):
                return False
            if(not '?' in item):
                loc=s.find(item,start)
                if(loc!=-1):
                    start=loc+len(item)
                else:
                    return False
            else:

                flag=True
                #if(start>=len(s)):
                    #return False
                print start,item,s
                for k in xrange(start,len(s)-len(item)+1):
                    if(xmatch(s[k:k+len(item)],item)):
                        flag=False
                        start=k+len(item)
                        break
                if(flag):
                    return False
        return True
"""
#45 Jump game
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

#46-47 Permutation
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

#48 Rotate image
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        print(n)
        modify={}
        for row in range(n):
            for col in range(n):
                if (n-1-col,row) not in modify:
                    #print(row,col)
                    modify[(row,col)], matrix[row][col] = matrix[row][col], matrix[n-1-col][row]
                else:
                    print(row,col)
                    matrix[row][col] = modify[(n-1-col,row)]
        print(modify)

sol = Solution()
matrix = [[1,2,3],[4,5,6],[7,8,9]]
a = [1, 2, 3]
print(a[-1])
matrix = matrix[::-1]
#sol.rotate(matrix)
print(matrix)
#len(matrix)

def rotate(self, matrix):
    n = len(matrix)
    for l in xrange(n / 2):
        r = n - 1 - l
        for p in xrange(l, r):
            q = n - 1 - p
            cache = matrix[l][p]
            matrix[l][p] = matrix[q][l]
            matrix[q][l] = matrix[r][q]
            matrix[r][q] = matrix[p][r]
            matrix[p][r] = cache

def rotate(self, matrix):
    n = len(matrix)
    matrix.reverse()
    for i in xrange(n):
        for j in xrange(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


#49 Group anagrams
class Solution(object):
    def groupAnagrams(self, strs):
        ans = collections.defaultdict(list)
        for s in strs:
            ans[tuple(sorted(s))].append(s)
        return ans.values()

class Solution:
    def groupAnagrams(strs):
        ans = collections.defaultdict(list)
        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord('a')] += 1
            ans[tuple(count)].append(s)
        return ans.values()

#51_N_queens:
class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        self.size = n
        self.build_board()
        res = []
        self.solve(res)
        return res

    def build_board(self):
        n = self.size
        self.board = ["."*n for _ in range(n)]
        return None

    def set_Q(self, i, j, char):
        s = self.board[i]
        n = self.size
        self.board[i] = s[0:j] + char + s[j+1:n]

    def is_safe_row(self,row):
        n = self.size
        for i in range(n):
            s = self.board[row]
            if self.board[row][i] == 'Q':
                return False
        return True

    def is_safe_col(self,col):
        n = self.size
        for i in range(n):
            if self.board[i][col] == 'Q':
                return False
        return True

    def is_safe_diagonal(self,row, col):
        n = self.size
        for i in range(n):
            j1 = i-row+col
            j2 = row+col - i
            if 0<=j1<n and self.board[i][j1] == 'Q':
                return False
            if 0<=j2<n and self.board[i][j2] == 'Q':
                return False
        return True

    def is_safe(self, row, col):
        return self.board[row][col] != 'Q' and self.is_safe_row(row) and self.is_safe_col(col) and self.is_safe_diagonal(row, col)

    def get_number_of_queens(self):
        count = 0
        n = self.size
        for row in range(n):
            for col in range(n):
                if self.board[row][col] == 'Q':
                    count += 1
        return count
    def find_safe_row(self):
        n = self.size
        for i in range(n):
            if self.is_safe_row(i):
                return i
        return None

    def find_safe_col(self, row):
        n = self.size
        for col in range(n):
            if self.is_safe(row,col):
                return col
        return None

    def solve(self, res):
        n_queens = self.get_number_of_queens()
        n = self.size
        if n_queens == n:
            res.append(self.board)
        for i in range(n):
            if self.is_safe_row(i):
                #print(i)
                j = self.find_safe_col(i)
                if j is not None:
                    #print('find j', j)
                    self.set_Q(i,j,'Q')
                    #print(self.board)
                    if self.solve(res):
                        self.set_Q(i,j,'.')
                        return True
                    self.set_Q(i,j,'.')
        return False


class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        self.size = n
        res = []
        self.reset_board()
        self.solve(res, 0, n)
        return res

    def reset_board(self):
        n = self.size
        self.board = ["."*n for _ in range(n)]
        return None

    def set_Q(self, i, j, char):
        s = self.board[i]
        n = self.size
        self.board[i] = s[0:j] + char + s[j+1:n]

    def is_safe_row(self,row):
        n = self.size
        for i in range(n):
            s = self.board[row]
            if self.board[row][i] == 'Q':
                return False
        return True

    def is_safe_col(self,col):
        n = self.size
        for i in range(n):
            if self.board[i][col] == 'Q':
                return False
        return True

    def is_safe_diagonal(self,row, col):
        n = self.size
        for i in range(n):
            j1 = i-row+col
            j2 = row+col - i
            if 0<=j1<n and self.board[i][j1] == 'Q':
                return False
            if 0<=j2<n and self.board[i][j2] == 'Q':
                return False
        return True

    def is_safe(self, row, col):
        return self.board[row][col] != 'Q' and self.is_safe_row(row) and self.is_safe_col(col) and self.is_safe_diagonal(row, col)

    def solve(self, res, row, n):
        if row >= n:
            return
        for i in range(n):
            if self.is_safe(row, i):
                self.set_Q(row,i,'Q')
                if row == n-1:
                    print(self.board)
                    res.append(self.board)
                    self.set_Q(row, i, '.')
                    return
                self.solve(res, row +1, n)
                self.set_Q(row, i,'.')



class Solution(object):
    def solveNQueens(self, n):
        res = []
        for i in range(n):
            self._solve(res, [i], n)
        return res

    def _solve(self, res, cols, n):
        m = len(cols)
        if m == n:
            res.append(self._make_grid(cols))
        else:
            for i in range(n):
                for j, col in enumerate(cols):
                    if abs(i-col) == m-j or i==col:
                        break
                else:
                    cols.append(i)
                    self._solve(res, cols, n)
                    cols.pop()

    def _make_grid(self, cols):
        n = len(cols)
        return ["".join(["Q" if i == col else "." for i in range(n)]) for col in cols]


class Solution(object):
    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: int
        """
        self.count = 0
        for i in range(n):
            self._solve([i], n)
        return self.count

    def _solve(self, cols, n):
        m = len(cols)
        if m == n:
            self.count += 1
            print(self.count)
        else:
            for i in range(n):
                for j, col in enumerate(cols):
                    if abs(i-col) == m-j or i==col:
                        break
                else:
                    cols.append(i)
                    self._solve(cols, n)
                    cols.pop()

# 54 Spiral Matrix
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if not matrix:
            return []
        low = 0
        n_row = len(matrix)-1
        n_col = len(matrix[0])-1

        res = []
        while low<=min(n_row, n_col):
            print(low, n_row, n_col)
            for i in range(low, n_col+1):
                res.append(matrix[low][i])
            for i in range(low+1, n_row+1):
                res.append(matrix[i][n_col])
            if low<min(n_row, n_col):
                for i in range(n_col-1, low-1, -1):
                    res.append(matrix[n_row][i])
                for i in range(n_row-1, low, -1):
                    res.append(matrix[i][low])
            low+=1
            n_col-=1
            n_row-=1
        return res

#54 Spiral matrix
class Solution(object):
    def spiralOrder(self, matrix):
        if not matrix: return []
        R, C = len(matrix), len(matrix[0])
        seen = [[False] * C for _ in matrix]
        ans = []
        dr = [0, 1, 0, -1]
        dc = [1, 0, -1, 0]
        r = c = di = 0
        for _ in range(R * C):
            ans.append(matrix[r][c])
            seen[r][c] = True
            cr, cc = r + dr[di], c + dc[di]
            if 0 <= cr < R and 0 <= cc < C and not seen[cr][cc]:
                r, c = cr, cc
            else:
                di = (di + 1) % 4
                r, c = r + dr[di], c + dc[di]
        return ans

class Solution(object):
    def spiralOrder(self, matrix):
        def spiral_coords(r1, c1, r2, c2):
            for c in range(c1, c2 + 1):
                yield r1, c
            for r in range(r1 + 1, r2 + 1):
                yield r, c2
            if r1 < r2 and c1 < c2:
                for c in range(c2 - 1, c1, -1):
                    yield r2, c
                for r in range(r2, r1, -1):
                    yield r, c1

        if not matrix: return []
        ans = []
        r1, r2 = 0, len(matrix) - 1
        c1, c2 = 0, len(matrix[0]) - 1
        while r1 <= r2 and c1 <= c2:
            for r, c in spiral_coords(r1, c1, r2, c2):
                ans.append(matrix[r][c])
            r1 += 1; r2 -= 1
            c1 += 1; c2 -= 1
        return ans

# 55 Jump game
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        n = len(nums)
        dp = [False] * n
        dp[n-1] = True
        temp_ind = n-1
        for i in range(n-2,-1,-1):
            if i+nums[i]>=temp_ind:
                temp_ind = i
                dp[i]=True

        return dp[0]


#56 Merge Intervals
# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        intervals.sort(key = lambda x: x.start)
        merge = []

        for interval in intervals:
            if merge==[] or merge[-1].end < interval.start:
                merge.append(interval)
            else:
                merge[-1].end = max(merge[-1].end, interval.end)
        return merge

#57 Spiral Matrix II
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        res = [[0]*n for _ in range(n)]
        dr = [0, 1, 0, -1]
        dc = [1, 0, -1, 0]
        r = c = di = 0
        for v in range(n*n):
            res[r][c] = v + 1
            #print(res)
            cr, cc = r+dr[di], c+dc[di]
            if 0<=cr<n and 0<=cc<n and res[cr][cc] == 0:
                r, c = cr, cc
            else:
                di = (di+1) % 4
                r, c = r+dr[di], c+dc[di]
        return res

def generateMatrix(self, n):
    A, lo = [], n*n+1
    while lo > 1:
        lo, hi = lo - len(A), lo
        A = [range(lo, hi)] + zip(*A[::-1])
    return A

def generateMatrix(self, n):
    A = [[n*n]]
    while A[0][0] > 1:
        A = [range(A[0][0] - len(A), A[0][0])] + zip(*A[::-1])
    return A * (n>0)

def generateMatrix(self, n):
    A = [[0] * n for _ in range(n)]
    i, j, di, dj = 0, 0, 0, 1
    for k in xrange(n*n):
        A[i][j] = k + 1
        if A[(i+di)%n][(j+dj)%n]:
            di, dj = dj, -di
        i += di
        j += dj
    return A

# 57 Insert Interval
# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """
        res = []
        for interval in intervals:
            if not newInterval or interval.end<newInterval.start:
                res.append(interval)
            elif newInterval.end<interval.start:
                res.append(newInterval)
                res.append(interval)
                newInterval = None
            else:
                newInterval.start = min(newInterval.start, interval.start)
                newInterval.end = max(newInterval.end, interval.end)
        if newInterval:
            res.append(newInterval)
        return res

#60 Permutation sequence
import math
class Solution(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        nums = [i for i in range(1,n+1)]
        return self.permutation(nums,k-1)

    def permutation(self, nums, k):
        n = len(nums)
        if k==0 or len(nums)==1:
            return "".join([str(i) for i in nums])
        if k == math.factorial(n)-1:
            return "".join([str(i) for i in nums[::-1]])
        ind = k//math.factorial(n-1)
        res = k % math.factorial(n-1)
        nums1 = nums[0:ind]+nums[ind+1:]

        ans = str(nums[ind]) + self.permutation(nums1, res)

        return ans

# 61 Rotate list
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if not head:
            return []
        if k == 0:
            return head
        pre, pre.next = self, head
        temp = head
        n = 0
        while temp:
            temp = temp.next
            n += 1
        if n == 1:
            return head
        k = k%n
        for _ in range(n-1):
            pre = pre.next
        temp = pre.next
        pre.next = None
        self.next = temp
        temp.next = head
        return self.rotateRight(self.next, k-1)

class Solution(object):
    def rotateRight(self, head, k):
        n, pre, current = 0, None, head
        while current:
            pre, current = current, current.next
            n += 1

        if not n or not k % n:
            return head

        tail = head
        for _ in xrange(n - k % n - 1):
            tail = tail.next

        next, tail.next, pre.next = tail.next, None, head
        return next

#62 Unique paths
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        dp = [[0]*m for _ in range(n)]

        for row in range(n):
            dp[row][0] = 1
        for col in range(m):
            dp[0][col] = 1
        for row in range(1,n):
            for col in range(1,m):
                dp[row][col] = dp[row-1][col] + dp[row][col-1]
        return dp[n-1][m-1]

#63: Unique paths II
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        n = len(obstacleGrid)
        m = len(obstacleGrid[0])
        dp = [[0]*m for _ in range(n)]
        for row in range(n):
            if obstacleGrid[row][0] == 0:
                dp[row][0] = 1
            else:
                break
        for col in range(m):
            if obstacleGrid[0][col] == 0:
                dp[0][col] = 1
            else:
                break
        for row in range(1,n):
            for col in range(1,m):
                dp[row][col] = dp[row-1][col] + dp[row][col-1]
                if obstacleGrid[row][col] == 1:
                    dp[row][col]  = 0
        return dp[n-1][m-1]

# 64 minimum path sum
class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        n = len(grid)
        m = len(grid[0])

        dp = [[0]*m for _ in range(n)]
        dp[0][0] = grid[0][0]
        for row in range(1, n):
            dp[row][0] = dp[row-1][0] + grid[row][0]
        for col in range(1, m):
            dp[0][col] = dp[0][col-1] + grid[0][col]
        for row in range(1,n):
            for col in range(1,m):
                dp[row][col] = min(dp[row-1][col],dp[row][col-1]) + grid[row][col]
        return dp[n-1][m-1]

# 65 Is number
class Solution(object):
    def isNumber(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if not s:
            return False
        s = s.strip()
        if s == '.':
            return False
        count_e = 0
        count_dot = 0
        count_digit = 0
        for i, char in enumerate(s):
            if not char.isdigit():
                if char == '.':
                    count_dot +=1
                    ind_dot = i
                    if count_dot >1:
                        return False
                elif char == 'e':
                    count_e += 1
                    if count_e >1:
                        return False
                    ind_e = i
                elif char == ' ':
                    if i>0 or i<len(s)-1:
                        return False
                elif char in ['-','+']:
                    if i>0 and s[i-1] != 'e':
                        return False
                else:
                    return False
            else:
                count_digit +=1
        if count_digit == 0:
            return False
        if count_e == 1:
            return self.isNumber(s[0:ind_e]) and\
                    self.isNumber(s[ind_e+1:]) and \
                    ('.' not in s[ind_e+1:]) and \
                    ind_e>0 and \
                    ind_e<len(s)-1

        return True
# 66 Plus one
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        s = ''.join(str(i) for i in digits)
        new_s = str(int(s) + 1)
        res = [0]*max(len(digits), len(new_s))
        j = len(new_s)-1
        for i in range(len(res)-1,len(res)-len(new_s)-1,-1):
            res[i] = int(new_s[j])
            j -= 1

        return res

# Add binary
class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        return bin(int(a, 2) + int(b,2))[2:]

#68 Text Justification
class Solution(object):
    def fullJustify(self, words, maxWidth):
        """
        :type words: List[str]
        :type maxWidth: int
        :rtype: List[str]
        """
        i = 0
        n_char = 0
        start = 0
        end = 0
        res = []
        while i<len(words):
            n_char += len(words[i])+1
            if n_char-1 > maxWidth:
                end = i
                res.append(self.joint_word(words[start:end], maxWidth))
                start=i
                n_char = 0
                i -= 1
            if i==len(words)-1:
                end = i+1
                res.append(self.joint_word(words[start:end], maxWidth, True))
            i += 1
        return res

    def joint_word(self, words, maxWidth, b_last_line = False):
        if not words:
            return ' '*maxWidth
        n_word = len(words)
        n_char = 0
        for word in words:
            n_char += len(word)+1 #add a space after each word
        n_char -= 1 # remove the last space after the last word
        if n_char > maxWidth:
            return False
        if n_word == 1:
            return words[0]+' '*(maxWidth-n_char)
        res = ''
        if not b_last_line:
            n_space_per_word = (maxWidth-n_char)//(n_word-1)
            n_space_left = (maxWidth-n_char)%(n_word-1)

            for i in range(n_word-1):
                if i<=n_space_left-1:
                    space_add = ' '*(n_space_per_word+2) #add two extra space, one for the
                    #default space and another one for unevenly distributed space
                else:
                    space_add = ' '*(n_space_per_word+1) #add one default space only
                res += words[i]+space_add
            res += words[-1]
        else:
            for word in words[0:-1]:
                res += word + ' '
            res += words[-1]   # add the last word
            res += ' '*(maxWidth-n_char)   # add all extra space
        return res

def fullJustify(self, words, maxWidth):
    res, cur, num_of_letters = [], [], 0
    for w in words:
        if num_of_letters + len(w) + len(cur) > maxWidth:
            for i in range(maxWidth - num_of_letters):
                cur[i%(len(cur)-1 or 1)] += ' '
            res.append(''.join(cur))
            cur, num_of_letters = [], 0
        cur += [w]
        num_of_letters += len(w)
    return res + [' '.join(cur).ljust(maxWidth)]

# 69 sqrt(x)
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x==0:
            return 0
        sqrt_x = (1 + x)//2
        while sqrt_x * sqrt_x > x:
            sqrt_x = (sqrt_x + x//sqrt_x)//2

        return sqrt_x

#70: climb stairs:
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 1:
            return 1
        dp = [0]*n
        dp[0] = 1
        dp[1] = 2
        for i in range(2,n):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n-1]

#71 simplify path
class Solution(object):
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        folder_strings = path.split('/')
        stack = []
        for char in folder_strings:
            if char == '' or char == '.':
                continue
            elif char == '..':
                if stack:
                    stack.pop()
            else:
                stack.append(char)
        print(stack)
        return "/"+"/".join(stack)

# 72 Edit distance
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        if not word1 and not word2:
            return 0
        dp = [[None]*(len(word2)+1) for _ in range(len(word1)+1)]
        for i in range(len(word1)+1):
            dp[i][0] = i
        for i in range(len(word2)+1):
            dp[0][i] = i
        for row in range(len(word1)+1):
            for col in range(len(word2)+1):
                if dp[row][col] == None:
                    if word1[row-1] == word2[col-1]:
                        dp[row][col] = dp[row-1][col-1]
                    else:
                        swap_last = dp[row-1][col-1]
                        add_last = dp[row-1][col]
                        sub_last = dp[row][col-1]
                        dp[row][col] = min(swap_last, add_last, sub_last) + 1
        return dp[-1][-1]

#73: set matrix to zeros
class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        zero_row, zero_col = set(), set()
        n = len(matrix)
        m = len(matrix[0])
        for i in range(n):
            for j in range(m):
                if matrix[i][j]==0:
                    zero_row.add(i)
                    zero_col.add(j)
        for i in zero_row:
            matrix[i] = [0]*m
        for j in zero_col:
            for i in range(n):
                matrix[i][j] = 0
"""
The idea is simple: if matrix[i][j] is equal to 0, then we set the elements
in the row i whose column index is less than j to zero, and set the elements
in row i whose column index is larger than j and value is not zero to a "tag"
such as None, indicating this element will be set to zero eventually but it's
not 0 originally. We can do the same thing to the elementes in column j.
In the following loop, we just have to focus on the elements whose values is 0
and do the same thing. In the end, replacing the None with 0.
"""

class Solution:
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        m = len(matrix)
        n = len(matrix[0])
        for i in range(0, m):
            for j in range(0, n):
                if matrix[i][j] == 0:
                    for ii in range(0, n):
                        if ii < j:
                            matrix[i][ii] = 0
                        elif ii > j and matrix[i][ii] != 0:
                            matrix[i][ii] = None
                    for jj in range(0, m):
                        if jj < i:
                            matrix[jj][j] = 0
                        elif jj > i and matrix[jj][j] != 0:
                            matrix[jj][j] = None
        for i in range(0, m):
            for j in range(0, n):
                if matrix[i][j] is None:
                    matrix[i][j] = 0

#74 Search a 2D matrix
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if matrix == [[]] or matrix == []:
            return False
        m = len(matrix)
        n = len(matrix[0])
        upper_bound = [] # get the last values of each row
        for i in range(m):
            upper_bound.append(matrix[i][-1])
        if target > upper_bound[-1]:
            return False
        row_ind = self.binary_search(upper_bound, target)

        col_ind = self.binary_search(matrix[row_ind], target)

        return target == matrix[row_ind][col_ind]

    def binary_search(self, nums, target):
        # return index that nums[index] = target
        # if not found that return index that
        # nums[index-1] < target < nums[index]
        if nums[0] > target:
            return 0
        if nums[-1] < target:
            return False
        l = 0
        r = len(nums) - 1
        while l <= r:
            mid = (l+r)//2
            if nums[mid] == target:
                return mid

            elif nums[mid] < target:
                l = mid+1
            else:
                r = mid-1
        return l

def searchMatrix(self, matrix, target):
    n = len(matrix[0])
    lo, hi = 0, len(matrix) * n
    while lo < hi:
        mid = (lo + hi) / 2
        x = matrix[mid/n][mid%n]
        if x < target:
            lo = mid + 1
        elif x > target:
            hi = mid
        else:
            return True
    return False

# 75 sorts colors - Dutch national flag algorithm
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        red, white, blue = 0, 0, len(nums)-1
        while white <= blue:
            if nums[white] == 0:
                nums[red], nums[white] = nums[white], nums[red]
                red += 1
                white += 1
            elif nums[white] == 1:
                white += 1
            else:
                nums[white], nums[blue] = nums[blue], nums[white]
                blue -= 1



# Minimum window substring
class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        if len(s)<len(t):
            return ''
        dist_t = {}
        dist_s = {}
        for char in t:
            dist_t[char] = dist_t.get(char, 0) + 1
        start, end = 0, 0
        count = 0
        solution = ''
        min_length = len(s)+1
        while end < len(s):
            char = s[end]
            dist_s[char] = dist_s.get(char, 0) + 1
            if dist_t.get(char,0) > 0 and dist_s[char]<=dist_t[char]:
                count += 1
            if count == len(t):
                start = self.remove_extra_char(dist_t, dist_s, s, start, end)
                #print(start)
                #print(end)
                if end-start+1 < min_length:
                    solution = s[start:end+1]
                    min_length = end-start+1
            end += 1
        return solution
    def remove_extra_char(self, dist_t, dist_s,s, start, end):
        ind = start
        while ind <= end:
            char = s[ind]
            if dist_t.get(char,0) == 0 or dist_s[char]>dist_t[char]:
                ind += 1
                dist_s[char] -= 1
            else:
                return ind
        return ind

def minWindow(self, s, t):
    need, missing = collections.Counter(t), len(t)
    i = I = J = 0
    for j, c in enumerate(s, 1):
        missing -= need[c] > 0
        need[c] -= 1
        if not missing:
            while i < j and need[s[i]] < 0:
                need[s[i]] += 1
                i += 1
            if not J or j - i <= J - I:
                I, J = i, j
    return s[I:J]

#77 combinations:
class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        nums = [x for x in range(1,n+1)]
        return self.permute(nums, k)
    def permute(self, nums, k):
        if not nums:
            return []
        if k==1:
            return [[x] for x in nums]
        res = []
        for i in range(len(nums)):
            temp = nums[i+1:]
            res+=([nums[i]]+v for v in self.permute(temp, k-1))
        return res


class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        if k > n:
            return []
        if k == 0:
            return []
        if k== 1:
            return [[x] for x in range(1,n+1)]
        res = []
        for i in range(n,-1,-1):
            res+=([i]+v for v in self.combine(i-1, k-1))
        return res

class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        if k > n:
            return []
        if k == 0:
            return []
        if k== 1:
            return [[x] for x in range(1,n+1)]
        if k == n:
            return [[i for i in range(1,n+1)]]
        res = []
        for i in range(n,-1,-1):
            res+=([i]+v for v in self.combine(i-1, k-1))
        return res

#78: Subsets
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if not nums:
            return [[]]

        res = [[]]
        for i in range(1,len(nums)+1):
            permute_nums = self.permute(nums, i)
            res += (x for x in permute_nums)

        return res

    def permute(self, nums, k):
        if not nums:
            return []
        if k==1:
            return [[x] for x in nums]
        if k==len(nums):
            return [nums]
        res = []
        for i in range(len(nums)):
            temp = nums[i+1:]
            res+=([nums[i]]+v for v in self.permute(temp, k-1))
        return res

class Solution(object):

    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums, lst = sorted(nums), [[]]
        for n in nums:
            for j in range(len(lst)):
                lst.append(lst[j]+ [n])
        return lst

# 79 Word Search:
class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        if board == []:
            return False
        self.board = board
        self.checked = {}
        m = len(board)
        n = len(board[0])
        for row in range(m):
            for col in range(n):
                self.checked[(row,col)] = False
        for row in range(m):
            for col in range(n):
                if board[row][col] == word[0]:
                    if self.check_neighbor(word, row, col):
                        return True
        return False

    def check_neighbor(self, word, start_row, start_col):
        if word == '':
            return True
        if start_row < 0 or start_row > len(self.board)-1 or start_col < 0 or start_col > len(self.board[0])-1:
            return False
        # print(start_row, start_col, word)
        if self.board[start_row][start_col] != word[0]:
            return False
        elif not self.checked[(start_row, start_col)]:
            self.checked[(start_row, start_col)] = True
            if self.check_neighbor(word[1:],start_row-1, start_col) or self.check_neighbor(word[1:],start_row+1, start_col)  \
               or self.check_neighbor(word[1:],start_row, start_col+1) or self.check_neighbor(word[1:],start_row, start_col-1):
                self.checked[(start_row, start_col)] = False
                return True
            self.checked[(start_row, start_col)] = False
        return False


# 80 remove duplicates from sorted array
class Solution:
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i = 0
        while i < len(nums):
            cur = i
            while i + 1 < len(nums) and nums[i+1] == nums[cur]:
                i += 1
            for j in range(cur+2, i+1):
                nums[j] = 'x'
            i += 1

        n_x = 0

        for i in range(len(nums)):
            if nums[i] == 'x':
                n_x += 1
                continue
            nums[i-n_x] = nums[i]
        return len(nums) - n_x
#solution nay cua minh ma k work
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        if len(nums)==1:
            return 1
        pointer1, pointer2, pointer3, count = -1, -1, 0, 1
        change_value = False
        cur = nums[0]
        while pointer3 < len(nums)-1:
            pointer3 += 1
            if nums[pointer3] == cur:
                count +=1
            if count>2 and pointer1<0:
                pointer1 = pointer3
            if (count>2 or nums[pointer3]!=cur) and pointer2>pointer1:
                #print('pointer1,2,3', pointer1, pointer2, pointer3)
                for i in range(pointer3-pointer2):
                    nums[pointer1], nums[pointer2] = nums[pointer2], nums[pointer1]
                    pointer1 += 1
                    pointer2 += 1
                    #print(nums)

                pointer2+=1

            if nums[pointer3] != cur:
                pointer2 = pointer3
                cur = nums[pointer2]
                count = 1

        return pointer1
# 81 search in rotated sorted array II (with possible duplicates)
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        if not nums:
            return False
        if nums[-1] == target:
            return True
        low, high = 0, len(nums)-1
        while low <= len(nums)-1 and nums[low] ==nums[-1]:
            low+= 1
        while high>=0 and nums[high] == nums[-1]:
            high -= 1
        while low <= high:
            mid = (low+high)//2
            if nums[mid] == target:
                return True
            if nums[low] <= nums[mid]:
                if nums[low]<=target<nums[mid]:
                    high = mid-1
                else:
                    low = mid+1
            else:
                if nums[mid] < target <=nums[high]:
                    low = mid+1
                else:
                    high = mid-1
        return False

#82: Remove duplicates from sorted list II:
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        pre, pre.next = self, head
        temp = pre
        cur=None
        while pre.next:
            #print('pre value', pre.next.val)
            if pre.next.next and pre.next.val==pre.next.next.val:
                cur = pre.next.val
                #print('cur',cur)
            if pre.next.val == cur:
                while pre.next and pre.next.val == cur:
                    if pre.next.next:
                        pre.next = pre.next.next
                    else:
                        pre.next = None
            else:
                pre = pre.next
        return temp.next

#83: Remove duplicates from sorted list :
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        pre, pre.next = self, head
        temp = pre
        while pre.next:
            #print('pre value', pre.next.val)
            if pre.next.next and pre.next.val==pre.next.next.val:
                pre.next = pre.next.next
            else:
                pre = pre.next
        return temp.next

#84: Largest rectangle in histogram
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        if not heights:
            return 0
        stack = []
        temp  = [0]
        for i, h in enumerate(heights):
            if stack and h<stack[-1][1]:
                (position, height) = stack.pop()
                temp.append(height*(i-position))
            else:
                stack.append((i,h))
        return max(temp)

class Solution(object):
    def largestRectangleArea(self, heights):
        temp = []
        stack = []
        heights.append(float('-inf'))
        for i,h in enumerate(heights):
            cpos = i
            while stack and h < stack[-1][0]:
                ch, cpos = stack.pop()
                temp.append(ch*(i - cpos ))
            stack.append([h, cpos])
        return max(temp) if temp else 0


# 85: Maximal rectangle
class Solution(object):
    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if len(matrix) == 0 or matrix == [[]]:
            return 0
        n = len(matrix[0])
        height = [0]*(n+1)
        #height[n] = len(matrix)+1
        res = 0

        for row in matrix:
            for i in range(n):
                height[i] = height[i] + 1 if row[i] == '1' else 0
            stack = []
            #print(height)
            for i_pos, h in enumerate(height):
                current_pos = i_pos
                while stack and h < stack[-1][0]:
                    current_h, current_pos = stack.pop()
                    res = max(res, current_h*(i_pos-current_pos))
                    #print(res)
                stack.append([h, current_pos])

        return res

# 86 partion list
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        if not head:
            return None
        head_less = less = ListNode(0)
        head_more = more = ListNode(0)
        while head:
            if head.val < x:
                less.next = head
                less = head
            else:
                more.next = head
                more = head

            head = head.next

        more.next = None
        less.next = head_more.next

        return head_less.next

#87 Scramble string
class Solution(object):
    def isScramble(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        if s1 == s2:
            return True

        if len(s1) != len(s2) or sorted(s1) != sorted(s2):
            return False

        for i in range(len(s1)-1):
            if (self.isScramble(s1[0:i+1], s2[0:i+1]) and self.isScramble(s1[i+1:], s2[i+1:])) or \
            (self.isScramble(s1[0:i+1], s2[-i-1:]) and self.isScramble(s1[i+1:], s2[0:-i-1])):
                return True
        return False


#88 Merge sorted array

class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        if m==0:
            nums1[0:n] = nums2
            return
        if n==0:
            return
        if nums2[n-1] < nums1[0]:
            nums1[0:m+n] = nums2 + nums1[0:m]
            return
        if nums1[m-1] < nums2[0]:
            nums1[0:m+n] = nums1[0:m] + nums2
            return
        if nums1[m-1]<nums2[n-1]:
            nums1[m+n-1] = nums2[n-1]
            self.merge(nums1, m, nums2[:n-1], n-1)
        else:
            nums1[m+n-1] = nums1[m-1]
            self.merge(nums1, m-1, nums2, n)

#89 Gray code
class Solution(object):
    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        if n == 0:
            return [0]
        temp = self.grayCode(n-1)
        res = temp[:]
        for num in temp[::-1]:
            res.append(2**(n-1)+num)
        return res

# 90 Subsets II
class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums, lst, length = sorted(nums), [[]], [1]
        for i in range(len(nums)):
            if i>0 and nums[i] == nums[i-1]:
                for j in range(length[-2], length[-1]):
                    lst.append(lst[j]+ [nums[i]])

            else:
                for j in range(len(lst)):
                    lst.append(lst[j]+ [nums[i]])

            length.append(len(lst))
        return lst

#91 Decode Ways
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        if len(s) == 1:
            if s[0] == "0":
                return 0
            return 1
        if len(s) == 2:
            if (int(s) > 26 and s[1] != '0') or int(s)==10 or int(s) == 20:
                return 1
            if 10 < int(s) <= 26:
                return 2
            return 0

        dp = [0]*len(s)
        dp[0] = self.numDecodings(s[0])
        dp[1] = self.numDecodings(s[0:2])

        for i in range(2, len(s)):
            if s[i] == '0':
                if s[i-1] == '0' or int(s[i-1])>2:
                    dp[i] = 0
                else:
                    dp[i] = dp[i-2]

            elif int(s[i-1:i+1]) > 26 or int(s[i-1:i+1]) < 10:
                dp[i] = dp[i-1]
            else:
                dp[i] = dp[i-2]+dp[i-1]

        #print(dp)
        return dp[-1]


# an other solution
def numDecodings(self, s):
"""
:type s: str
:rtype: int
"""
    if not s or s[0]=='0': return 0
    p=q=1
    D=set(map(str,range(1,27)))
    for i in range(1,len(s)):
        q,p=(s[i] in D)*q+(s[i-1:i+1] in D)*p,q
    return q

#92: reverse between singly-linked list
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        current = head
        Reverse = None
        for i in range(1, n + 1):
            temp = current.next
            if i == m - 1:              # example: 1->2->3->4->5->None
                first = current         # first: 1
                end = current.next      # end: 2
            if i >= m:
                current.next = Reverse  # reverse linked list
                Reverse = current
            current = temp

        if m == 1:                      # when m at begin
            head.next = current
            return Reverse

        first.next = Reverse            # connect reverse linked list
        end.next = current

        return head


class Solution(object):
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        if not head:
            return None

        first_tail, second_tail = head, head

        while pos<n:
            if pos<m:
                first_tail = first_tail.next
            second_tail = second_tail.next
            pos +=1

        second_head = first_tail.next
        third_head = second_tail.next
        first_tail.next = None
        second_tail.next = None
        self.reversed_list(second_head)
        first_tail.next = second_tail
        second_head.next = third_head
    def reversed_list(self, head):
        if not head:
            return
        if head.next == None:
            return

        pre, pre.next, cur = self, head, head
        while cur.next:
            cur = cur.next
            pre = pre.next

        pre.next = None
        temp_head = head.next
        head.next = None
        self.reversed_list(temp_head)
        cur.next = temp_head

#206 Reverse Linked List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return None
        if not head.next:
            return head
        cur = head
        reverse = None
        while cur.next:
            temp = cur.next
            cur.next = reverse
            reverse = cur
            cur = temp

        cur.next = reverse
        return cur

#93 Resetore IP addresses
class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        return self.IpAddressesPossible(s, 4)


    def IpAddressesPossible(self, s, k):
        if not s or len(s) < k or len(s) > 3*k:
            return []
        if k==1 and self.isValidPart(s) :
            return [s]
        res = []
        for i in range(1,4):
            if k-1 <= len(s)-i <= 3*(k-1) and self.isValidPart(s[0:i]):
                res+= (s[0:i]+"."+ v for v in self.IpAddressesPossible(s[i:], k-1))
        return res

    def isValidPart(self, s):
        if not s or len(s)>3:
            return False
        if s[0] == "0" and len(s)>1:
            return False
        if s=="0":
            return True
        if 0<int(s)<256:
            return True
        return False

#95 Unique Binary Search Trees II
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        if n==0:
            return []
        return self.get_all_trees(range(1, n+1))

    def get_all_trees(self, nums):
        if not nums:
            return [None]

        res = []
        for i in range(len(nums)):
            for left in self.get_all_trees(nums[0:i]):
                for right in self.get_all_trees(nums[i+1:len(nums)]):
                    tree = TreeNode(nums[i])
                    tree.left = left
                    tree.right = right
                    res.append(tree)
        return res

#100: Same Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if not p and not q:
            return True
        if not p and q:
            return False
        if p and not q:
            return False
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

#101 Symmetric Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        all_val, stack = [], [root]
        while stack:
            temp_val, temp_tree_node = [], []
            for x in stack:
                if isinstance(x, TreeNode):
                    temp_tree_node.append(x.left)
                    temp_tree_node.append(x.right)
                    temp_val.append(x.val)
                else:
                    temp_val.append(x)
            stack = temp_tree_node
            all_val.append(temp_val)
        for level in all_val:
            if level != level[::-1]:
                return False
        return True

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root: return True
        return self.helper(root.left, root.right)

    def helper(self, left, right):
        if left == right == None: return True
        if not left or not right: return False
        if left.val != right.val: return False
        return self.helper(left.left, right.right) and self.helper(left.right, right.left)

#102 Binary Tree Level Order Trversal:
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        all_val, stack = [], [root]
        while stack:
            temp_val, temp_tree_node = [], []
            for x in stack:
                if isinstance(x, TreeNode):
                    temp_tree_node.append(x.left)
                    temp_tree_node.append(x.right)
                    temp_val.append(x.val)
                elif x:
                    temp_val.append(x)
            print(temp_val)
            if temp_val:
                all_val.append(temp_val)
            stack = temp_tree_node
        return all_val
"""
Breadh First Search

Using BFS, at any instant only L1 and L1+1 nodes are in the queue.
When we start the while loop, we have L1 nodes in the queue.
for _ in range(len(q)) allows us to dequeue L1 nodes one by one and add L2 children one by one.
Time complexity: O(N). Space Complexity: O(N)
"""
from collections import deque
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        q, result = deque(), []
        if root:
            q.append(root)
        while len(q):
            level = []
            for _ in range(len(q)):
                x = q.popleft()
                level.append(x.val)
                if x.left:
                    q.append(x.left)
                if x.right:
                    q.append(x.right)
            result.append(level)
        return result
"""
Depth First Search

Use a variable to track level in the tree and use simple Pre-Order traversal
Add sub-lists to result as we move down the levels
Time Complexity: O(N)
Space Complexity: O(N) + O(h) for stack space
"""
class Solution(object):
    def levelOrder(self, root):
        result = []
        self.helper(root, 0, result)
        return result

    def helper(self, root, level, result):
        if root is None:
            return
        if len(result) <= level:
            result.append([])
        result[level].append(root.val)
        self.helper(root.left, level+1, result)
        self.helper(root.right, level+1, result)

#103 Binary Tree Zigzag level order traversal
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

from collections import deque
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        q, result, level_number = deque(), [], 0
        if root:
            q.append(root)
        while len(q):
            level = []
            for _ in range(len(q)):
                x = q.popleft()
                level.append(x.val)
                if x.left:
                    q.append(x.left)
                if x.right:
                    q.append(x.right)
            if level_number%2 == 1:
                level = level[::-1]
            level_number+= 1
            result.append(level)
        return result

#104: maximum depth of binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from collections import deque
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        tree_depth, de_que = 0, deque()
        de_que.append(root)
        while de_que:
            has_level = False
            for _ in range(len(de_que)):
                x = de_que.popleft()
                if x:
                    has_level = True
                if x.left:
                    de_que.append(x.left)
                if x.right:
                    de_que.append(x.right)
            if has_level:
                tree_depth += 1
        return tree_depth

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right)) if root else 0

#105: Construct Binary Tree from Preorder and Inorder Traversal
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if not preorder and not inorder:
            return None
        root = TreeNode(preorder[0])
        ind = inorder.index(root.val)
        root.left = self.buildTree(preorder[1:ind+1], inorder[0:ind])
        root.right = self.buildTree(preorder[ind+1:], inorder[ind+1:])
        return root

#106: Construct Binary Tree from Postorder and Inorder Traversal
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if not postorder and not inorder:
            return None
        root = TreeNode(postorder[-1])
        ind = inorder.index(root.val)
        root.left = self.buildTree(inorder[0:ind], postorder[0:ind])
        root.right = self.buildTree(inorder[ind+1:], postorder[ind:-1])
        return root

#107: Binary Tree Level Order Trversal II
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from collections import deque
class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        q, result = deque(), []
        if root:
            q.append(root)
        while len(q):
            level = []
            for _ in range(len(q)):
                x = q.popleft()
                level.append(x.val)
                if x.left:
                    q.append(x.left)
                if x.right:
                    q.append(x.right)
            result.append(level)
        return result[::-1]

# dfs recursively
def levelOrderBottom1(self, root):
    res = []
    self.dfs(root, 0, res)
    return res

def dfs(self, root, level, res):
    if root:
        if len(res) < level + 1:
            res.insert(0, [])
        res[-(level+1)].append(root.val)
        self.dfs(root.left, level+1, res)
        self.dfs(root.right, level+1, res)

# dfs + stack
def levelOrderBottom2(self, root):
    stack = [(root, 0)]
    res = []
    while stack:
        node, level = stack.pop()
        if node:
            if len(res) < level+1:
                res.insert(0, [])
            res[-(level+1)].append(node.val)
            stack.append((node.right, level+1))
            stack.append((node.left, level+1))
    return res

# bfs + queue
def levelOrderBottom(self, root):
    queue, res = collections.deque([(root, 0)]), []
    while queue:
        node, level = queue.popleft()
        if node:
            if len(res) < level+1:
                res.insert(0, [])
            res[-(level+1)].append(node.val)
            queue.append((node.left, level+1))
            queue.append((node.right, level+1))
    return res

#108 Convert sorted array to Binary search Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return None
        n = len(nums)
        root = TreeNode(nums[n//2])
        root.left = self.sortedArrayToBST(nums[0:n//2])
        root.right = self.sortedArrayToBST(nums[n//2+1:])
        return root

#109 Convert sorted link list to Binary search Tree
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        if not head:
            return None
        if not head.next:
            return TreeNode(head.val)
        cur, length = head, 0
        while cur.next:
            length+=1
            cur = cur.next

        cur, temp_length = head, 0
        if length>=2:
            left = head
        else:
            left = None
        right = None
        while cur.next:
            temp = cur.next
            if temp_length == length//2-1:
                cur.next = None

            elif temp_length == length//2:
                mid = cur.val
                right = cur.next
            cur = temp
            temp_length+=1

        root = TreeNode(mid)
        root.left = self.sortedListToBST(left)
        root.right = self.sortedListToBST(right)
        return root

#Q 97 Interleaving string
class Solution(object):
    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        if len(s3)!=len(s1) + len(s2):
            return False
        dp = [[""]*(len(s2)+1) for _ in range(len(s1)+1)]

        for i in range(len(s1)+1):
            for j in range(len(s2)+1):
                if i==0 and j==0:
                    dp[i][j] = True
                elif i==0:
                    dp[i][j] = dp[i][j-1] and s2[j-1] == s3[i+j-1]
                elif j==0:
                    dp[i][j] = dp[i-1][j] and s1[i-1] == s3[i+j-1]
                else:
                    dp[i][j] = dp[i-1][j] and s1[i-1] == s3[i+j-1] or dp[i][j-1] and s2[j-1] == s3[i+j-1]

        return dp[-1][-1]

#Q99 Recover Binary Search Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def recoverTree(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        self.pre, self.first, self.second = None, None, None
        self.inOrderTranversal(root)
        self.first.val, self.second.val = self.second.val, self.first.val

    def inOrderTranversal(self, root):
        """
        This is actually an inOrderTranversal problem. We will visit each node and
        every time we found that the previous node.val > current node.val we set them
        as first and second "bad" node
        """
        if not root:
            return None
        self.inOrderTranversal(root.left)
        if self.pre and self.pre.val > root.val:
            if not self.first:
                self.first = self.pre

            self.second = root
        self.pre = root
        self.inOrderTranversal(root.right)

#Q110 Balanced Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        if abs(self.height(root.left)-self.height(root.right)) > 1:
            return False
        return self.isBalanced(root.left) and self.isBalanced(root.right)

    def height(self, root):
        if not root:
            return 0
        return max(self.height(root.left), self.height(root.right))+1

#Q111 Minimum Depth of Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        if not root.left and not root.right:
            return 1
        elif not root.left:
            return self.minDepth(root.right) + 1
        elif not root.right:
            return self.minDepth(root.left) + 1
        else:
            return min(self.minDepth(root.left), self.minDepth(root.right))+1

#Q112: Path Sum
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if not root and sum is None:
            return True
        if not root and sum is not None:
            return False
        if not root.left and not root.right:
            return root.val == sum
        return self.hasPathSum(root.left, sum-root.val) or self.hasPathSum(root.right, sum-root.val)

#113 Path Sum II
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        res = []
        def pathSumAll(root, sum, ans):
            if not root:
                return
            if root.left is None and root.right is None:
                if root.val == sum:
                    ans.append(sum)
                    res.append(ans)
                    return
            if root.left:
                pathSumAll(root.left, sum-root.val, ans+[root.val])
            if root.right:
                pathSumAll(root.right, sum-root.val, ans+[root.val])

        pathSumAll(root, sum, [])
        return res

class Solution:
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        if root:
            if root.left and root.right:
                return ([[root.val]+i for i in self.pathSum(root.left, sum-root.val)]+
                        [[root.val]+i for i in self.pathSum(root.right, sum-root.val)])
            if root.left:
                return [[root.val]+i for i in self.pathSum(root.left, sum-root.val)]
            if root.right:
                return [[root.val]+i for i in self.pathSum(root.right, sum-root.val)]
            if root.val == sum:
                return [[root.val]]
        return []

#114 Flatten Binary Tree to Linked List:
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):

    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """


        if not root:
            return None

        self.pre = None
        self.preOrder(root)
    def preOrder(self, root):
        temp1, temp2 = root.left, root.right
        if self.pre:
            self.pre.left = None
            self.pre.right = root
        self.pre = root
        if temp1:
            self.preOrder(temp1)
        if temp2:
            self.preOrder(temp2)

#115 Distinct Subsequences
class Solution(object):
    def numDistinct(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        if len(s) < len(t):
            return 0
        dp = [[0]*(len(t) + 1) for _ in range(len(s) +1)]
        for i in range(0, len(s)+1):
            dp[i][0] = 1

        for i in range(1, len(s)+1):
            for j in range(1, len(t)+1):
                if i<j:
                    dp[i][j] = 0
                elif i==j:
                    dp[i][j] = 1 if s[:i] == t[:j] else 0
                else:
                    dp[i][j] = dp[i-1][j-1] * int(s[i-1] == t[j-1]) + dp[i-1][j]
        print(dp)

        return dp[-1][-1]

#116, 117. Populating next right pointers in each node
# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

from collections import deque
class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        q = deque()
        if root:
            q.append(root)
            #root.next = None
        while len(q):
            pre_node = None
            temp_length = len(q)
            for i in range(temp_length):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
                if pre_node:
                    pre_node.next = node

                pre_node = node


#118 Pascal's Triangle
class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if numRows == 0:
            return []
        if numRows == 1:
            return [[1]]
        res = [[1]]
        for row in range(1, numRows):
            temp = [1]*(row+1)
            pre_row = res[-1]
            for i in range(1, row):
                temp[i] = pre_row[i] + pre_row[i-1]
            res.append(temp)
        return res

#119 Pascal's Triangle II
class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """

        if rowIndex == 0:
            return [1]

        pre_row = [1]
        for row in range(1, rowIndex+1):
            temp = [1]*(row+1)
            for i in range(1, row):
                temp[i] = pre_row[i] + pre_row[i-1]
            pre_row = temp
        return temp

#120 Triangle
class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        if triangle == [[]]:
            return 0
        num_row = len(triangle)
        if num_row == 1:
            return triangle[0][0]
        dp = triangle[0]
        for row in range(1, num_row):
            temp = [0]*(row+1)
            last_row = dp
            current_row = triangle[row]
            temp[0] = current_row[0]+dp[0]
            temp[-1] = current_row[-1] + dp[-1]
            for i in range(1, row):
                temp[i] = current_row[i] + min(last_row[i], last_row[i-1])
            dp = temp
        return min(dp)

#121 Best time to buy and sell stock
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) < 2:
            return 0
        profit = 0
        minimum = prices[0]
        for i in range(1, len(prices)):
            price = prices[i]
            profit = max(profit, price-minimum)
            minimum = min(minimum, price)
        return profit

#122 Best time to buy and sell stock II
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) < 2:
            return 0
        profit = 0
        cur_min = prices[0]
        for i in range(1, len(prices)):
            if prices[i] < prices[i-1]:
                profit += prices[i-1] - cur_min
                cur_min = prices[i]
        if prices[-1] > cur_min:
            profit += prices[-1] - cur_min
        return profit

#123 Best time to buy and sell stock III
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) < 2:
            return 0
        i = 0
        #find all the valleys and peaks
        peaks_valleys = []
        while i < len(prices)-1:
            while i < len(prices)-1 and prices[i] >= prices[i+1]:
                i += 1
            peaks_valleys.append(prices[i]) #add valleys

            while i< len(prices)-1 and prices[i] <= prices[i+1]:
                i += 1
            peaks_valleys.append(prices[i]) #add peaks
        # get the maximum profit by looping over two event
        if len(peaks_valleys) == 2:
            return peaks_valleys[1]-peaks_valleys[0]

        profit = peaks_valleys[1]-peaks_valleys[0]

        for i in range(0, len(peaks_valleys), 2):
            max_profit_pre_event = self.max_1_transaction_profit(peaks_valleys[0:i])
            max_profit_after_event = self.max_1_transaction_profit(peaks_valleys[i:])

            profit = max(profit, max_profit_pre_event+ max_profit_after_event)

        return profit

    def max_1_transaction_profit (self, prices):
        if len(prices) < 2:
            return 0
        profit = 0
        minimum = prices[0]
        for i in range(1, len(prices)):
            price = prices[i]
            profit = max(profit, price-minimum)
            minimum = min(minimum, price)
        return profit

# sol 2:
class Solution(object):
    def maxProfit(self, prices):

        if len(prices)<=1: return 0

        # O(n) counting from left, find the max gain up to each day (not ending at each day)
        left = [0]*len(prices)
        curmin = prices[0]
        for i in range(1, len(prices)):
            curmin = min(curmin, prices[i])
            left[i] = max(prices[i]-curmin, left[i-1])

        # O(n) counting from right
        right = [0]*len(prices)
        curmax = prices[-1]
        for i in range(len(prices)-2, -1, -1):
            curmax = max(curmax, prices[i])
            right[i] = max(curmax-prices[i], right[i+1])

        # O(n)
        max2t = 0
        for i in range(len(prices)):
            max2t = max(max2t, left[i] + right[i])
                return max2t
# best solution
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        dp[k, i] = max(dp[k, i-1], prices[i] - (prices[j] - dp[k-1, j-1])) for j=0,..,i

        need to find the min of (prices[j] - dp[k-1, j-1]) for all j < i
        we name this min_potential_lost
        """
        if not prices:
            return 0
        dp=[0] * 3
        min_potential_lost = [prices[0]] * 3
        for i in range(1, len(prices)):
            for k in range(1, 3):
                min_potential_lost[k] = min(min_potential_lost[k], prices[i] - dp[k-1])
                dp[k] = max(dp[k], prices[i] - min_potential_lost[k])
        return dp[2]

#188 Best time to buy and sell stock IV
class Solution(object):
    def maxProfit(self, k, prices):
        """
        :type k: int
        :type prices: List[int]
        :rtype: int
        dynamic programing:
        dp[k, i] = max(dp[k, i-1], prices[i] - (prices[j] - dp[k-1, j-1])) for j=0,..,i
        need to find the min of (prices[j] - dp[k-1, j-1]) for all j < i
        we name this min_potential_lost
        """
        if not prices:
            return 0
        n = len(prices)
        if k>=n/2:
            s=0
            for i in range(1,n):
                if prices[i]>prices[i-1]:
                    s+=prices[i]-prices[i-1]
            return s

        dp=[None] * (k+1)
        dp[0] = 0
        min_potential_lost = [prices[0]] * (k+1)
        for i in range(1, len(prices)):
            for k in range(1, k+1):
                min_potential_lost[k] = min(min_potential_lost[k], prices[i] - dp[k-1])
                dp[k] = max(dp[k], prices[i] - min_potential_lost[k])
        return dp[-1]

#124: Binary Tree Maximum path Sum
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        self.res = root.val
        a = self.max1BranchSum(root)
        return self.res
    def max1BranchSum(self, root):
        """
        this one will return the maximum branch (either left or right of a root)
        """
        if not root:
            return 0
        max_left = self.max1BranchSum(root.left)
        max_right = self.max1BranchSum(root.right)
        max_sum = max(max_left, max_right, 0) + root.val
        self.res = max(self.res, max_sum, max_left + max_right + root.val)
        return max_sum

#sol 2:
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        self.res = root.val
        from_root, no_root = self.twoMaxPath(root)
        return max(from_root, no_root)
    def twoMaxPath(self, root):
        """
        this one will return two max values: one go from the root and the other didn't from the root
        """
        if not root:
            return float("-inf"), float("-inf")
        if not root.left and not root.right:
            return root.val, float("-inf")

        from_root_left, no_root_left = self.twoMaxPath(root.left)
        from_root_right, no_root_right = self.twoMaxPath(root.right)
        from_root = max(from_root_left + root.val, from_root_right + root.val, root.val)
        no_root = max(no_root_left, no_root_right, from_root_left, from_root_right, from_root_left+root.val+from_root_right)
        return from_root, no_root

#183 SQL Customers who never orders
# Write your MySQL query statement below
SELECT Customers.name AS 'Customers' from Customers
WHERE id NOT IN (SELECT CustomerId from Orders)
#595 SQL Big Countries
# Write your MySQL query statement below
SELECT name, population, area from world
WHERE population>25000000 OR area>3000000

#125 Valid Palindrome
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if not s:
            return True
        l, r = 0, len(s)-1
        while l<r:
            while not s[l].isalnum() and l<r:
                l+=1
            while not s[r].isalnum() and r>l:
                r-=1
            if l<r and s[l].lower() != s[r].lower():
                return False
            l+=1
            r-=1
        return True



# 126 word ladder (starting points for 2019)
class Solution(object):
    def findLadders(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: List[List[str]]
        """
        if endWord not in wordList:
            return []

        level = {}
        level[beginWord] = [[beginWord]]
        res = []
        wordList = set(wordList)

        while level:
            next_level = collections.defaultdict(list)
            for u in level:
                if u == endWord:
                    res.extend(k for k in level[u])
                    return res
                else:
                    # make all posible transformation
                    for i in range(len(u)):
                        for c in 'abcdefghijklmnopqrstuvwxyz':
                            new_word = u[:i] + c + u[i+1:]
                            if new_word in wordList:
                                next_level[new_word] += [x + [new_word] for x in level[u]]

            wordList -= set(next_level.keys())
            level = next_level

        return res


# 129:
# 130: Surrounded Regions
class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        if board == []:
            return
        queue = collections.deque([])

        n_r = len(board)
        n_c = len(board[0])
        for r in range(n_r):
            for c in range(n_c):
                if ( (r in [0, n_r-1]) or (c in [0, n_c-1]) ) and board[r][c] == 'O':
                    queue.append((r, c))
        while queue:
            (r, c) = queue.popleft()
            if (0<=r<n_r) and (0<=c<n_c) and board[r][c] == 'O':
                board[r][c] = 'D' # mark this one as Dont Change (D)
                queue.append((r-1, c))
                queue.append((r, c-1))
                queue.append((r+1, c))
                queue.append((r, c+1))

        for r in range(n_r):
            for c in range(n_c):
                if board[r][c] == 'O':
                    board[r][c] = 'X'
                elif board[r][c] == 'D':
                    board[r][c] = 'O'
#200: Number of Islands:
class Solution(object):
    def numIslands(self, grid):
        if not grid:
            return 0

        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    self.dfs(grid, i, j)
                    count += 1
        return count

    def dfs(self, grid, i, j):
        if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]) or grid[i][j] != '1':
            return
        grid[i][j] = '#'
        self.dfs(grid, i+1, j)
        self.dfs(grid, i-1, j)
        self.dfs(grid, i, j+1)
        self.dfs(grid, i, j-1)

# 131 Palindrome partioning
class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        res = []
        self.dfs(s, [], res)
        return res

    def dfs(self, s, path, res):
        if not s:
            res.append(path)
            return
        for i in range(1, len(s)+1):
            if self.is_palindrome(s[:i]):
                self.dfs(s[i:], path + [s[:i]], res)

    def is_palindrome(self, s):
        return s == s[::-1]

# 132 Palindrome partioning II

class Solution(object):
    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s is None or len(s)==1 or s==s[::-1]:
            return 0

        for i in range(1, len(s)):
            if s[:i] == s[:i][::-1] and s[i:] == s[i:][::-1]:
                return 1

        min_cut=[0] # starting value for i = 0

        is_palindrome = [[False] * len(s) for _ in s]

        for i in range(1, len(s)):
            min_min_cut = float('inf')
            for j in range(i, -1, -1):

                # check if s[j:i+1] is palindrome:
                if ((j==i) or (j==i-1 and s[i]==s[j]) or (s[i]==s[j] and is_palindrome[j+1][i-1])):
                    is_palindrome[j][i] = True
                    if j==0:
                        cur_min_cut = 0
                    else:
                        cur_min_cut = min_cut[j-1] + 1
                    min_min_cut = min_min_cut if min_min_cut < cur_min_cut else cur_min_cut
            min_cut.append(min_min_cut)
        return min_cut[len(s)-1]

# 133   clone graph
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, neighbors):
        self.val = val
        self.neighbors = neighbors
"""
class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        if node is None:
            return None
        stack = []
        clone = {}
        clone[node] = Node(node.val, [])
        stack.append(node)

        while stack:
            temp_node = stack.pop()
            for nb in temp_node.neighbors:
                if nb not in clone:
                    clone[nb] = Node(nb.val, [])
                    stack.append(nb)
                clone[temp_node].neighbors.append(clone[nb])
        return clone[node]

# 134 Gas station
class Solution:
    def canCompleteCircuit(self, gas, cost):

        st, mx, s = -1, -1, 0
        for i in range(len(gas)-1, -1, -1):
            s += gas[i] - cost[i]
            if s > mx:
                mx, st = s, i
        return st if s >= 0 else -1
# 135 candy
class Solution(object):
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        if len(ratings) <= 1:
            return len(ratings)
        up, down, slope, next_slope, candy = 0, 0, 0, 0, 0

        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i-1]:
                next_slope = 1
            elif ratings[i] < ratings[i-1]:
                next_slope = -1
            else:
                next_slope = 0

            if  (slope > 0 and next_slope ==0) or (slope <0 and next_slope >=0):
                # print(i, up, down, slope, next_slope, candy)
                candy += self.count(up) + self.count(down) + max(up, down)
                up = 0
                down = 0

            if next_slope > 0:
                up += 1
            elif next_slope < 0:
                down += 1
            else:
                candy += 1

            slope = next_slope
        candy += self.count(up) + self.count(down) + max(up, down) + 1
        return candy
    def count(self, x):
        return x*(x+1)//2

# 136 single number
from collections import Counter
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        freq = Counter(nums)
        for key in freq.keys():
            if freq[key] == 1:
                return key

class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        a = 0
        for i in nums:
            a ^= i
        return a

class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return 2 * sum(set(nums)) - sum(nums)

# 137 single number II
class Solution(object):
    def singleNumber(self, nums):
        a = b = 0
        for n in nums:
            b = (b^n)&~a
            a = (a^n)&~b
        return b

# 138. copy list with random pointer
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, next, random):
        self.val = val
        self.next = next
        self.random = random
"""
class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        if head is None:
            return
        stack = []
        clone = {}
        clone[head] = Node(head.val, None, None )
        stack.append(head)

        while stack:
            temp_node = stack.pop()
            n = temp_node.next
            r = temp_node.random
            if n:
                if n not in clone:
                    clone[n] = Node(n.val, None, None)
                    stack.append(n)
                clone[temp_node].next = clone[n]

            if r:
                if r not in clone:
                    clone[r] = Node(r.val, None, None)
                    stack.append(r)
                clone[temp_node].random = clone[r]

        return clone[head]

#139 Word break
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        if s in wordDict:
            return True
        dp = [False] * (len(s)+1)
        dp[0] = True
        for i in range(1, len(s)+1):
            for j in range(i):
                if s[j:i] in wordDict and dp[j]:
                    dp[i] = True
                    break

        return dp[-1]

class Solution(object):
    def wordBreak(self, s, wordDict):
        for word in wordDict:
            self.insert(word)
        self.search(s)
        if self.count > 0:
            return True
        else:
            return False

    # Trie build up
    def __init__(self):
        self.root={}
        self.count = 0

    def insert(self, word):
        p = self.root
        for c in word:
            if c not in p.keys():
                p[c] = {}
            p = p[c]
        p['#'] = True

    # search function
    def search(self, word):
        memo = []
        return self.find(word, 0, len(word), memo)

    def find(self, prefix, start, end, memo):
        p = self.root
        if start == end:
            self.count += 1
        for index, c in enumerate(prefix[start:]):
            # one solution possible, end process
            if self.count > 0:
                break
            # word not in trie, break up
            if c not in p.keys():
                break
            # record index success, reduce recursion
            if start + index in memo:
                p=p[c]
                continue
            if '#' in p[c]:
                memo.append(start+index)
                self.find(prefix, start+index+1, end, memo)
            p=p[c]

class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        lookBackLen = set()
        for w in wordDict:
            lookBackLen.add(len(w))
        wordDict = set(wordDict)

        d = [False] * len(s)
        for i in range(len(s)):
            for b in lookBackLen:
                if s[i+1-b:i+1] in wordDict and (i-b < 0 or d[i-b]):
                    d[i] = True

        return d[-1]

#140 word break II
class Solution(object):
    def wordBreak(self, s, wordDict):
        self.lens = sorted(list(set([len(word) for word in wordDict])))
        self.dic = {l: set() for l in self.lens}
        for word in wordDict:
            self.dic[len(word)].add(word)
        return self.find(s, {})

    def find(self, s, memo):
        if s in memo:
            return memo[s]
        res = []
        sL = len(s)
        for l in self.lens:
            ch = s[:l]
            if ch in self.dic[l]:
                if l < sL:
                    for word in self.find(s[l:], memo):
                        res.append(ch + ' ' + word)
                else:
                    res.append(ch)
                    break
        memo[s] = res
        return res

# 141 linked list cycle:
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head is None:
            return False
        slow = head
        fast = head.next
        while slow != fast:
            if fast is None or fast.next is None:
                return False
            slow=slow.next
            fast = fast.next.next
        return True

# 142 Linked list cycle II:
    # Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: node
        """
        if head is None:
            return None
        node_meet = self.isCycle(head)
        if node_meet is None:
            return None

        cur = head
        while node_meet != cur:
            node_meet = node_meet.next
            cur = cur.next
        return node_meet
    def isCycle(self, head):
        """
        detect if there is circle. return the node where fast meet slow
        :type head: ListNode
        :rtype: node
        """
        slow = fast = head
        while (fast is not None) and (fast.next is not None):
            slow=slow.next
            fast = fast.next.next
            if slow == fast:
                return slow
        return None

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """

        if not head:    return head

        low = fast = head
        while fast and fast.next:
            low = low.next
            fast = fast.next.next

        tmp = low.next
        low.next = None

        # reverse the right half
        tail = cur = tmp
        while cur and cur.next:
            cur = cur.next
            tail.next = cur.next
            cur.next = tmp
            tmp = cur
            cur = tail

        # insert in order
        mid = tmp
        while mid:
            tmp = mid.next
            mid.next = head.next
            head.next = mid
            head = head.next.next
            mid = tmp

# 144 Binary Tree Preorder Traversal:
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root is None:
            return []
        stack = [root]
        out_put = []

        while stack:
            node = stack.pop()
            if node.val:
                out_put.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return out_put

# 146 LRU Cacche
class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.key_list = {} # contains key values pairs
        self.start_idx = 0 # current index of start of list
        self.list = [] # contains keys in ordered of used
        self.current_size = 0 # current size of cache
        self.key_list_idx = {} # index where the key is stored in the list/array

    def re_order_list(self, key):
        if self.key_list_idx[key] == len(self.list) - 1: # if the key is the most recent used
            return
        self.list[self.key_list_idx[key]] = None
        self.key_list_idx[key] = len(self.list)
        self.list.append(key)
        return
    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.key_list:
            return -1
        else:
            self.re_order_list(key)
            return self.key_list[key]

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if self.capacity < 1:
            return
        if key in self.key_list:
            self.key_list[key] = value
            self.re_order_list(key)
            return
        if self.current_size == self.capacity:
            while self.list[self.start_idx] is None:
                self.start_idx += 1
            remove_el = self.list[self.start_idx]
            self.key_list.pop(remove_el, None)
            self.key_list_idx.pop(remove_el, None)
            self.start_idx += 1
        else:
            self.current_size += 1
        self.key_list[key] = value
        self.key_list_idx[key] = len(self.list)
        self.list.append(key)



# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

# 147. Insertion sort list
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def insertionSortList(self, head):
        p = dummy = ListNode(0)
        cur = dummy.next = head
        while cur and cur.next:
            val = cur.next.val
            if cur.val < val:
                cur = cur.next
                continue
            if p.next.val > val:
                p = dummy
            while p.next.val < val:
                p = p.next
            new = cur.next
            cur.next = new.next
            new.next = p.next
            p.next = new
        return dummy.next

class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        length, h = 0, head
        while h:
            length += 1
            h = h.next
        return self.merge_sort_list(head, length)

    def merge(self, head1, head2):
        dummy = merge_head=ListNode(0)
        while head1 and head2:
            if head1.val < head2.val:
                merge_head.next = head1
                head1 = head1.next
            else:
                merge_head.next = head2
                head2 = head2.next
            merge_head = merge_head.next
        merge_head.next = head1 or head2
        return dummy.next

    def merge_sort_list(self, head, length):
        if length <= 1:
            return head
        mid, ct = head, 1
        while ct < length//2:
            mid = mid.next
            ct += 1
        second_half = mid.next
        mid.next = None
        l1 = self.merge_sort_list(head, length//2)
        l2 = self.merge_sort_list(second_half, length - length//2)
        return self.merge(l1, l2)

#149 Max points on a line
#150 Evaluate reverse plish notation
class solution(object):
    def evalRPN(self, tokens):
            """
            :type tokens: List[str]
            :rtype: int
            """
            stack = []

            for x in tokens:
                if x in ['+', '-', '*', '/']:
                    num1 = stack.pop()
                    num2 = stack.pop()
                    if x=='+':
                        num = num1 + num2
                    elif x=='-':
                        num = num2 - num1
                    elif x=='*':
                        num = num1*num2
                    elif x=='/':
                        num = int(num2/float(num1))
                    stack.append(num)
                else:
                    stack.append(int(x))
            return stack[0]

# 151
# 152 Maximum product subarray
class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        if not nums:
            return 0

        n = len(nums)-1

        max_up_to = [0] * (n+1)
        max_up_to[0] = nums[0]

        min_up_to = [0] * (n+1)
        min_up_to[0] = nums[0]

        for i in range(1, n+1):
            max_up_to[i] = max(max_up_to[i-1] * nums[i],
                               min_up_to[i-1] * nums[i],
                               nums[i])
            min_up_to[i] = min(max_up_to[i-1] * nums[i],
                               min_up_to[i-1] * nums[i],
                               nums[i])

        return max(max_up_to)

class Solution(object):
    def maxProduct(self, A):
        B = A[::-1]
        for i in range(1, len(A)):
            A[i] *= A[i - 1] or 1
            B[i] *= B[i - 1] or 1
        return max(A+B)

# 153 Find minimum in rotated sorted array
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if len(nums) < 3:
            return min(nums)
        if(nums[0] < nums[-1]):
            return nums[0]
        mid = (len(nums)-1) // 2
        if nums[0] < nums[mid]:
            return self.findMin(nums[mid+1:])
        else:
            return self.findMin(nums[:mid+1])

class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1:
            return nums[0]
        idx = 0

        while idx<len(nums)-2 and nums[idx] < nums[idx+1]:
            idx += 1
        return min(nums[0], nums[idx+1], nums[-1])
# 154 find minimum in roated sorted array II
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return None
        if len(nums) == 1:
            return nums[0]

        if nums[0] < nums[-1]:
            return nums[0]

        left, right = 0, len(nums)-1

        while left < right:

            mid = (left + right)//2

            if nums[mid] > nums[right]:
                left = mid + 1
            elif nums[mid] < nums[right]:
                right = mid
            else:
                right -= 1

        return nums[left]

# 155 Min stack:
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.data = []
        self.min = None

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        if self.min is None:
            self.min = x
        else:
            self.min = min(x, self.min)
        self.data.append(x)

    def pop(self):
        """
        :rtype: None
        """
        x = self.data[-1]
        self.data.pop()
        if self.min == x and self.data != []:
            self.min = min(self.data)
        elif self.min ==x and self.data == []:
            self.min = None
    def top(self):
        """
        :rtype: int
        """
        return self.data[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return self.min

#160: Intersection of Two linked lists
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        p1, p2 = headA, headB
        while p1 is not p2:
            p1 = headB if p1 is None else p1.next
            p2 = headA if p2 is None else p2.next
        return p1

# 162 Find Peak Element
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return None
        return self.search_peak(nums,0, len(nums)-1)
    def search_peak(self, nums, l, r):
        if l==r:
            return l
        mid = (l+r)//2
        if nums[mid]>nums[mid+1]:
            return self.search_peak(nums,l, mid)
        return self.search_peak(nums, mid+1, r)

# 164 Maxium Gap
#Pigeonhole buckets
class Bucket:
    def __init__(self):
        self.isUsed = False
        self.minVal = float("inf")
        self.maxVal = -float("inf")

class Solution:
    def maximumGap(self, nums):
        n = len(nums)

        if n < 2:
            return 0
        minNum = min(nums)
        maxNum = max(nums)

        bucketSize = max(1, (maxNum - minNum) // (n - 1))
        bucketNum = (maxNum - minNum) // bucketSize + 1

        buckets = [Bucket() for _ in range(bucketNum)]

        for num in nums:
            bucketIdx = (num - minNum) // bucketSize
            buckets[bucketIdx].isUsed = True
            buckets[bucketIdx].minVal = min(buckets[bucketIdx].minVal, num)
            buckets[bucketIdx].maxVal = max(buckets[bucketIdx].maxVal, num)

        res = 0
        prevEnd = None
        for bucket in buckets:
            if not bucket.isUsed: continue

            if prevEnd != None:
                res = max(res, bucket.minVal - prevEnd)
            prevEnd = bucket.maxVal

        return res
#165 Compare version numbers:
class Solution(object):
    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """

        version1 = map(int, version1.split('.'))
        version2 = map(int, version2.split('.'))

        if len(version1) > len(version2):
            version2 += [0] * (len(version1) - len(version2))
        else:
            version1 += [0] * (len(version2) - len(version1))

        for i in range(len(version1)):
            if version1[i] > version2[i]:
                return 1
            elif version1[i] < version2[i]:
                return -1

        return 0

#166
class Solution(object):
    def fractionToDecimal(self, numerator, denominator):
        """
        :type numerator: int
        :type denominator: int
        :rtype: str
        """
        if numerator % denominator == 0:
            return str(numerator//denominator)
        res = []
        if numerator * denominator < 0:
            res.append('-')
        numerator, denominator = abs(numerator), abs(denominator)
        mod, remainder = divmod(numerator, denominator)
        res.append(str(mod))
        res.append('.')
        dict = {}
        idx = len(res)-1
        while remainder != 0:
            if remainder not in dict:
                idx += 1
                dict[remainder] = idx
                mod, remainder = divmod(remainder*10, denominator)
                res.append(str(mod))
            else:
                res.insert(dict[remainder], '(')
                res.append(')')
                break
        return "".join(res)

# 167 Two sum II- input array is sorted
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        left = 0
        right = len(numbers) - 1
        while left < right:
            if numbers[left] + numbers[right] == target:
                return [left+1, right+1]
            elif numbers[left] + numbers[right] > target:
                right -= 1
            else:
                left += 1
        return []

#168: Excel sheet column title:
class Solution(object):
    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """
        Alphabet = {0:"Z"}
        for i in range(25):
            Alphabet[i+1] = chr(i+65)

        res = []

        while n > 0:
            n, rem = divmod(n, 26)
            if rem==0:
                n -= 1
            res.append(Alphabet[rem])

        return "".join(res[::-1])

#169: Majority Element
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        counts = collections.Counter(nums)
        return max(counts.keys(), key=counts.get)

#170:
#171: Excel sheet column number
class Solution(object):
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        dict = {"Z":26}
        for i in range(25):
            dict[chr(i+65)] = i+1
        n = len(s)
        res = 0
        for i, char in enumerate(s):
            res += 26**(n-1-i)*dict[char]
        return res

#172: factorial trailing zeroes:
class Solution(object):
    def trailingZeroes(self, n):
        res = 0
        base = 5
        while base <= n:
            res += n // base
            base *= 5
        return res

import math
class Solution:
    def trailingZeroes(self, n):
        assert n >= 0, n
        zeros = 0
        q = n

        while q:
            q //= 5
            zeros += q
        # print(zeros)
        return zeros

#173 Binary search tree iterator
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class BSTIterator(object):

    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.root = root
        self.all_nums = []
        self.inorder_traversal(root)
        self.current_idx = -1
        self._len = len(self.all_nums)
    def next(self):
        """
        @return the next smallest number
        :rtype: int
        """
        self.current_idx += 1
        return self.all_nums[self.current_idx]


    def hasNext(self):
        """
        @return whether we have a next smallest number
        :rtype: bool
        """
        return self.current_idx + 1  < self._len
    def inorder_traversal(self, node):
        """
        perform inorder traversal of binary Tree
        """
        if node is None:
            return
        if node.left:
            self.inorder_traversal(node.left)
        self.all_nums.append(node.val)
        if node.right:
            self.inorder_traversal(node.right)


# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()
#174 Dungeon Game;
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        ''' This is a derterministic Markov Decision Process (MPD),
            we can write the following Bellman Equation
            V(i, j) = max(min(V(i + 1, j), V(i, j + 1)) - R(i, j), 1)
            let V(i, j) be the minimum HP needed if starting from
            position (i, j) and R(i,j) being the reward to be in
            position (i, j), then we can write the above Bellman Equation,

        '''
        m = len(dungeon)
        n = len(dungeon[0])
        V = [[1 for j in range(n)] for i in range(m)]
        V[m - 1][n - 1] = max(1 - dungeon[m - 1][n - 1], 1)
        for i in range(m - 2, -1, -1):
            V[i][n - 1] = max(V[i + 1][n - 1] - dungeon[i][n - 1], 1)
        for j in range(n - 2, -1, -1):
            V[m - 1][j] = max(V[m - 1][j + 1] - dungeon[m - 1][j], 1)
        for i in range(m - 2, -1, -1):
            for j in range(n - 2, -1, -1):
                V[i][j] = max(min(V[i + 1][j], V[i][j + 1]) - dungeon[i][j], 1)
        return V[0][0]

#175: Combine two tables:
# Write your MySQL query statement below
SELECT Person.FirstName, Person.LastName, Address.City, Address.State
FROM Person LEFT JOIN Address ON Person.PersonId =Address.PersonId

#176 Second highest salary
# Write your MySQL query statement below
# Write your MySQL query statement below
SELECT IFNULL((SELECT DISTINCT Salary
               from Employee
               Order by Salary DESC
               LIMIT 1 OFFSET 1), NULL) As SecondHighestSalary

#177 # getNthHighestSalary
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
    DECLARE M INT;
    SET M=N-1;
    RETURN (
      # Write your MySQL query statement below.


      SELECT DISTINCT Salary
            FROM EMployee
            ORDER BY Salary DESC
            LIMIT 1 OFFSET M
  );
END

#178 Rank Scores
# First one uses two variables, one for the current rank
# and one for the previous score.
SELECT
  Score,
  @rank := @rank + (@prev <> (@prev := Score)) Rank
FROM
  Scores,
  (SELECT @rank := 0, @prev := -1) init
ORDER BY Score desc

SELECT
  Score,
  (SELECT count(distinct Score) FROM Scores WHERE Score >= s.Score) Rank
FROM Scores s
ORDER BY Score desc

SELECT
  Score,
  (SELECT count(*) FROM (SELECT distinct Score s FROM Scores) tmp WHERE s >= Score) Rank
FROM Scores
ORDER BY Score desc

SELECT s.Score, count(distinct t.score) Rank
FROM Scores s JOIN Scores t ON s.Score <= t.score
GROUP BY s.Id
ORDER BY s.Score desc

#179 Largest number
class LargerNumKey(str):
    def __lt__(x, y):
        return x+y > y+x

class Solution(object):
    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        num_sort = sorted(map(str, nums), key=LargerNumKey)
        largest_num = ''.join(num_sort)
        return '0' if largest_num[0] == '0' else largest_num


#180 Consecutive Numbers:
# Write your MySQL query statement below
SELECT DISTINCT
    l1.Num AS ConsecutiveNums
FROM
    Logs l1,
    Logs l2,
    Logs l3
WHERE
    l1.Id = l2.Id - 1
    AND l2.Id = l3.Id - 1
    AND l1.Num = l2.Num
    AND l2.Num = l3.Num
;

select distinct num as consecutiveNums
from (select num,sum(c) over (order by id) as flag
    from (select id, num, case when LAG(Num) OVER (order by id)- Num = 0 then 0 else 1 end as c
from logs) a) b
group by num,flag
having count(*) >=3


#181 Employees Earnign more than their managers
# Write your MySQL query statement below
Select
    a.Name As 'Employee'
From
    Employee As a,
    Employee As b
Where
    a.ManagerID = b.Id
        AND a.Salary > b.Salary


SELECT
     a.NAME AS Employee
FROM Employee AS a JOIN Employee AS b
     ON a.ManagerId = b.Id
     AND a.Salary > b.Salary

#182 Duplicate email
select Email from
(
  select Email, count(Email) as num
  from Person
  group by Email
) as statistic
where num > 1
;

select Email
from Person
group by Email
having count(Email) > 1;

#184: Department Highest Salary
# Write your MySQL query statement below
Select
    Department.name As 'Department',
    Employee.name As 'Employee',
    Salary
From
    Employee
        Join
    Department ON Employee.DepartmentId = Department.Id
WHERE
    (Employee.DepartmentId, Salary) IN
    (Select
        DepartmentId, MAX(Salary)
     FROM
        Employee
     GROUP BY DepartmentId
    )

#185: Department Top Three Salaries
select tD.Name as 'Department', tE1.Name as 'Employee', tE1.Salary
from Employee as tE1
    Inner join Department as tD on tE1.DepartmentId = tD.Id
    Left join Employee as tE2 on tE1.DepartmentId = tE2.DepartmentId and tE1.Salary <= tE2.Salary
group by tE1.Id
having count(distinct tE2.Salary) <= 3
order by tE1.DepartmentId, tE1.Salary desc


#187 Repeated DNA sequences:
from collections import defaultdict, Counter
class Solution(object):
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        dict = defaultdict()
        for i in range(len(s)-9):
            sub = s[i:i+10]
            if sub in dict:
                dict[sub] += 1
            else:
                dict[sub] = 1
        counter = Counter(dict)
        res = [x for x in counter if counter[x]>1]
        return res

#188 Rotate array:
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k = k%n
        nums[:] = nums[n-k:n] + nums[0:n-k]

#190 Reverse Bits
class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        string = bin(n)[2:].zfill(32)
        resv = string[::-1]
        res = int(resv,2)
        return res

#191 Number of 1 Bits
from collections import Counter
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        s = str(bin(n)[2:])

        res = Counter([x for x in s])

        return res['1']

class Solutions(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        return str(bin(n)).count('1')

# 196 Delete Duplicate Emails
Delete p1 from Person p1, Person p2
Where
    p1.Email = p2.Email AND p1.Id > p2.Id


# 197: Rising temperature
# Write your MySQL query statement below
SELECT
    weather.id AS 'Id'
FROM
    weather
        JOIN
    weather w ON DATEDIFF(weather.RecordDate, w.RecordDate) = 1
        AND weather.Temperature > w.Temperature
;

#198: House Robber
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        n = len(nums)
        if n==1:
            return nums[0]
        dp = [None]*n
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        return dp[-1]
# 213 House Robber II
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0

        n = len(nums)
        if n==1:
            return nums[0]
        if n==2:
            return max(nums[0], nums[1])
        #nums = nums + [nums[0]]
        nums_1 = nums[0:n-1]
        nums_2 = nums[1:n]
        return max(self.rob_original(nums_1), self.rob_original(nums_2))
    def rob_original(self, nums):
        if not nums:
            return 0
        n = len(nums)
        if n==1:
            return nums[0]
        dp=[None]*n
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-1], dp[i-2]+nums[i])
        return dp[-1]

#199 Binary Tree Right Side View
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        rightmost_value_at_depth = dict() #depth -> node.val
        max_depth = -1
        stack = [(root, 0)]
        while stack:
            node, depth = stack.pop()
            if node is not None:
                #get the max_depth of the tree
                max_depth = max(max_depth, depth)

                #only insert into dict if depth is not already present
                rightmost_value_at_depth.setdefault(depth, node.val)
                stack.append((node.left, depth+1))
                stack.append((node.right, depth+1))
        return [rightmost_value_at_depth[depth] for depth in range(max_depth+1)]
# Note: in this case, if we use stack we will implement DFS; if we use
# queue, we will implement BFS

#201 Bitwise AND of Numbers Range
"""
The hardest part of this problem is to find the regular pattern.
For example, for number 26 to 30
Their binary form are:
11010
11011
11100　　
11101　　
11110

Because we are trying to find bitwise AND, so if any bit there are at least one 0 and one 1, it always 0. In this case, it is 11000.
So we are go to cut all these bit that they are different. In this case we cut the right 3 bit.

I think after understand this, the code is trivial:
"""
class Solution(object):
    def rangeBitwiseAnd(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        i = 0 # i means we have how many bits are 0 on the right
        while m != n:
            m >>= 1
            n >>= 1
            i += 1
        return m << i

# 202 Happy number
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        i = 0
        memo = {}
        while n != 1:
            temp = self.sum_square_digits(n)
            memo[n] = temp
            n = temp
            if n in memo:
                return False
        return True
    def sum_square_digits(self, x):
        str_x = str(x)
        str_x_digits = [int(c)**2 for c in str_x]
        return sum(str_x_digits)
# 203 Remove linked list elements
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head
        pre, cur = dummy, head
        while cur:
            if cur.val == val:
                while cur and cur.val ==val:
                    cur = cur.next
                pre.next = cur
            pre = pre.next
            if cur:
                cur = cur.next
        return dummy.next

#204 Count primes:
def countPrimes(self, n):
        if n < 3:
            return 0
        primes = [True] * n
        primes[0] = primes[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if primes[i]:
                primes[i * i: n: i] = [False] * len(primes[i * i: n: i])
        return sum(primes)

#205: Isomorphic Strings:
class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) != len(t):
            return False
        mapping = dict()
        for i, ch in enumerate(s):
            if ch not in mapping:
                if t[i] in mapping.values():
                    return False
                mapping.setdefault(ch, t[i])
            elif mapping[ch] != t[i]:
                return False
        return True

#207: Course Schedule:
"""
This is topological sorting.
https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm

"""
# Sol1 based on dfs:
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        graph = [[] for _ in range(numCourses)]
        visit = [0] * numCourses
        for course, pre in prerequisites:
            graph[course].append(pre)

        def visit_dfs(i):
            if visit[i] == -1:
                return False
            if visit[i] == 1:
                return True
            visit[i] = -1
            for course in graph[i]:
                if not visit_dfs(course):
                    return False
            visit[i] = 1
            return True

        for i in range(numCourses):
            if not visit_dfs(i):
                return False
        return True
# Sol2 based on Kahn's algorithm:
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        graph = {i: set() for i in range(numCourses)}
        indeg = {i: 0 for i in range(numCourses)}
        for s, e in prerequisites:
            graph[s] |= {e}
            indeg[e] += 1
        queue = [i for i in range(numCourses) if indeg[i] == 0]
        visits = set(queue)
        for node in queue:
            for neigh in graph[node]:
                if neigh in visits:
                    return False
                indeg[neigh] -= 1
                if indeg[neigh] == 0:
                    visits.add(neigh)
                    queue.append(neigh)
        return len(visits) == numCourses

# 208 Implement Trie (Prefix Tree)
class TrieNode(object):
    def __init__(self):
        self.children = [None]*26
        self.is_end_word = False

class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def _char_to_index(self, ch):
        """
        private helper function to
        convert key current character
        into index. Assume only use 'a'
        through 'z' and lower case
        """
        return ord(ch) - ord('a')

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: None
        """
        pointer = self.root
        length = len(word)
        for level in range(length):
            index = self._char_to_index(word[level])
            # if current character is not present:
            if pointer.children[index] is None:
                pointer.children[index] = TrieNode()
            pointer = pointer.children[index]
        # mark last node as leaf
        pointer.is_end_word = True



    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        pointer = self.root
        length = len(word)
        for level in range(length):
            index = self._char_to_index(word[level])
            # if current character is not present:
            if pointer.children[index] is None:
                return False
            pointer = pointer.children[index]
        return pointer != None and pointer.is_end_word


    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        pointer = self.root
        length = len(prefix)
        for level in range(length):
            index = self._char_to_index(prefix[level])
            # if current character is not present:
            if pointer.children[index] is None:
                return False
            pointer = pointer.children[index]
        return True

# 209. Minimum size subarray sum:
class Solution(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        ans = float('inf')
        left, sums = 0, 0
        for i in range(len(nums)):
            sums += nums[i]
            while sums >= s:
                ans = min(ans, i+1-left)
                sums -= nums[left]
                left += 1
        return ans if ans != float('inf') else 0


#210: Course Schedule II
class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        graph = {i: set() for i in range(numCourses)}
        in_coming_edge = {i: 0 for i in range(numCourses)}
        visit = [0] * numCourses
        for course, pre_req in prerequisites:
            graph[course] |= {pre_req}
            in_coming_edge[pre_req] += 1
        outside_edge = [i for i in range(numCourses) if in_coming_edge[i] == 0]
        visits = set(outside_edge)
        course_ordered = []
        while outside_edge:
            temp_course = outside_edge.pop()
            for neigh in graph[temp_course]:
                if neigh in visits:
                    return []
                in_coming_edge[neigh] -= 1
                if in_coming_edge[neigh] == 0:
                    outside_edge.append(neigh)
                    visits.add(neigh)
            course_ordered.append(temp_course)
        return course_ordered[::-1] if len(course_ordered) == numCourses else []

# Sol 2: DFS
class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        graph = {i: set() for i in range(numCourses)}
        visit = [0] * numCourses

        for s, e in prerequisites:
            graph[s] |= {e}


        course_ordered = []

        def dfs_visit(course):
            # print(course)
            if visit[course] == 1:
                return True
            if visit[course] == -1:
                # is_possible = False
                return False

            visit[course] = -1
            if course in graph:
                for node in graph[course]:
                    if not dfs_visit(node):
                        return False
            visit[course] = 1
            course_ordered.append(course)
            # print(course_ordered)
            return True
        for course in range(numCourses):
            if visit[course] == 0:
                if not dfs_visit(course):
                    return []
        return course_ordered if len(course_ordered) == numCourses else []


# 211 Add and search word - data structure design
class TrieNode(object):
    def __init__(self):
        self.children = [None]*26
        self.is_end_word = False

class WordDictionary(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def _char_to_index(self, ch):
        """
        private helper function to
        convert key current character
        into index. Assume only use 'a'
        through 'z' and lower case
        """
        return ord(ch) - ord('a')

    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: None
        """
        pointer = self.root
        length = len(word)
        for level in range(length):
            idx = self._char_to_index(word[level])
            if pointer.children[idx] is None:
                pointer.children[idx] = TrieNode()
            pointer = pointer.children[idx]
        pointer.is_end_word = True

    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        def find_word(trie_node, word):
            if trie_node is None:
                return False
            if not word:
                return trie_node.is_end_word
            if word[0] == '.':
                for child in trie_node.children:
                    if find_word(child, word[1:]):
                        return True
                return False
            else:
                index = self._char_to_index(word[0])
                child = trie_node.children[index]

                return find_word(child, word[1:])
                #return False

        return find_word(self.root, word)

class WordDictionary(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.val = False
        self.child = {}


    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: None
        """
        cur = self
        for c in word:
            if c not in cur.child:
                cur.child[c] = WordDictionary()
            cur = cur.child[c]
        cur.val = True


    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        stack,n = [(self,0)],len(word)
        while stack:
            cur,depth = stack.pop(0)
            if depth >= n: return False
            for k in cur.child:
                if k == word[depth] or word[depth] == ".":
                    stack.append((cur.child[k],depth+1))
                    if depth == n-1 and cur.child[k].val: return True
        return False


#212 Word search II:
# This solution is just an improvement of Word search I (Q79), however
# it will give LTE(Limit time error) because it takes a lot of time
# to re-checked the word with similar prefix. We must use Trie() to get
# away with that
class Solution(object):
    def findWords(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        self.board = board
        self.n_row = len(self.board)
        self.n_col = len(self.board[0])
        self.visit = [[0]*self.n_col for _ in range(self.n_row)]
        res = []
        for row in range(self.n_row):
            for col in range(self.n_col):
                char = self.board[row][col]
                for word in words:
                    if word[0]==char and word not in res:
                        if self.find_word(row, col, word, 0):
                            res.append(word)
        return res

    def find_word(self, row, col, word, k):
        if k>len(word)-1:
            return True
        if row < 0 or row > self.n_row-1 or col<0 or col>self.n_col-1:
            return False
        if self.board[row][col] != word[k]:
            return False

        if self.visit[row][col] == 0:
            self.visit[row][col] = 1

            if self.find_word(row-1, col, word, k+1) or self.find_word(row+1, col, word, k+1) or \
                self.find_word(row, col-1, word, k+1) or self.find_word(row, col+1, word, k+1):
                self.visit[row][col] = 0
                return True
            self.visit[row][col] = 0
        return False

# using Trie()
class TrieNode(object):
    def __init__(self):
        self.word = None
        self.children = {}

class Trie(object):
    def __init__(self):
        self.root = TrieNode()
    def insert(self, word):
        pointer = self.root
        for char in word:
            if char not in pointer.children:
                pointer.children[char] = TrieNode()
            pointer = pointer.children[char]
        pointer.word = word

class Solution(object):
    def search(self, row, col, root, board, n_row, n_col, res):
        """
        helper function to search word from row and col in board. the
        word is in the Trie (root). When the word is end (i.e. root.word
        is not None), root.word will be added to res
        """
        char = board[row][col]
        if not (char and char in root.children):
            return
        board[row][col], root = None, root.children[char]
        if root.word:
            res.append(root.word)
            root.word = None # removed it

        for x, y in ((0, -1), (-1, 0), (0, 1), (1, 0)):
            neigh_row, neigh_col = row +x, col + y
            if 0<= neigh_row < n_row and 0<=neigh_col<n_col:
                self.search(neigh_row, neigh_col, root, board, n_row, n_col, res)

        board[row][col] = char

    def findWords(self, board, words):
        if not board:
            return []
        trie = Trie()
        for word in words:
            trie.insert(word)
        res = []
        n_row = len(board)
        n_col = len(board[0])
        for row in range(n_row):
            for col in range(n_col):
                self.search(row, col, trie.root, board, n_row, n_col, res)
        return res

# 980 Unique Paths III:
class Solution(object):
    def uniquePathsIII(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if not grid:
            return 0
        n_row = len(grid)
        n_col = len(grid[0])
        self.number_visited_need = 0
        for row in range(n_row):
            for col in range(n_col):
                if grid[row][col] == 1:
                    start_row, start_col = row, col
                elif grid[row][col] == 0:
                    self.number_visited_need += 1
        # print(self.number_visited_need)
        self.res = 0
        def find_path(r, c, visit):
            # print(r, c, visit)
            if grid[r][c] == 2:
                #print(visit, self.number_visited_need)
                if visit == self.number_visited_need+2:
                    self.res += 1
                return
            grid[r][c] = -1
            for nr, nc in [(r, c-1), (r, c+1), (r-1, c), (r+1, c)]:
                if 0<=nr<n_row and 0<=nc<n_col and grid[nr][nc] % 2 == 0:
                    find_path(nr, nc, visit+1)
            grid[r][c] = 0
        # grid[start_row][start_col] = -1
        find_path(start_row, start_col, 1)
        return self.res

# 215 Kth Largest element in an array:

class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        heap = []
        for num in nums:
            heapq.heappush(heap, -num)
        for _ in range(k-1):
            heapq.heappop(heap)
        return -heapq.heappop(heap)

# 216 Combination Sum III:
class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        nums = [i for i in range(1, 10)]

        return self.combination_sum(k, n, nums)

    def combination_sum(self, k, n, nums):
        if k>len(nums) or sum(nums[-k:])<n or not nums:
            return []
        if k==1:
            if n in nums:
                return [[n]]
            else:
                return []
        res = []
        for i, num in enumerate(nums):
            # temp_nums = numsnums[i+1:]
            # print(temp_nums)
            remain = self.combination_sum(k-1, n-num, nums[i+1:])
            for x in remain:
                res.append([num] + x)
        return res

#214 Shortest Palindrome
class Solution(object):
    def shortestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if not s:
            return ''

        reverse = s[::-1]

        for i in range(len(reverse)):
            if s.startswith(reverse[i:]):
                return reverse[:i] + s

#sol 2 is related to KMP failure function
# KMP is findung the longest prefix that is also suffix
# KMP algorithm:
class Solution(object):
    def shortestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        new_s =  s + '#' + s[::-1]
        fail = self.compute_kmp_fail(new_s)
        i = fail[-1]
        return s[i:][::-1] + s


    def compute_kmp_fail(self, s):
        k = 0
        j = 1
        m = len(s)
        fail = [0]*m
        while j<m:
            if s[j] == s[k]:
                fail[j] = k+1
                j += 1
                k += 1
            elif k>0:
                k = fail[k-1]
            else:
                j += 1
        return fail

# 219 Contains DupblicateII
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        if len(set(nums))==len(nums):
            return False
        duplicate_list = {}
        for i, num in enumerate(nums):
            if num not in duplicate_list:
                duplicate_list[num] = i
            else:
                distance = i - duplicate_list[num]
                duplicate_list[num] = i
                if distance <= k:
                    return True
        return False
# 220 Contains dubplicate III
class Solution(object):
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        """
        # moving each number in nums into buckets of size t
        # maintain maximum number of buckets is k to make sure
        # the difference in index is k
        bucket = {}
        for i, num in enumerate(nums):
            # get which bucket it is in:
            buck_num = num//t if t!=0 else num
            # get the offset bucket which typical is 1
            # i.e. the no candidates must be in the same bucket
            # or two adjunct bucket. if t==0, we only check in the same bucket
            offset = 1 if t!=0 else 0
            for idx in range(buck_num-offset, buck_num+offset+1):
                if idx in bucket and abs(bucket[idx] - num) <= t:
                    return True
            bucket[buck_num] = num
            if len(bucket.keys()) > k:
                # remove the bucket associate with nums[i-k]
                # again we determin the buck_num for nums[i-k]
                # equal nums[i-k]//t or nums[i-k] if t==0
                buck_num_of_nums_i_k = nums[i-k] //t if t!=0 else nums[i-k]
                del bucket[buck_num_of_nums_i_k]
        return False

# 221 Maximum square:
class Solution(object):
    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if not matrix:
            return 0
        n_row = len(matrix)
        n_col = len(matrix[0])
        dp = [[0]*n_col for _ in range(n_row)]
        max_len = 0
        for i in range(n_row):
            for j in range(n_col):
                if matrix[i][j] == '1':
                    # print(i, j)
                    if i==0 or j==0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i-1][j-1],dp[i][j-1]) + 1

                    max_len = max(max_len, dp[i][j])

        return max_len*max_len

# 222 count complete tree nodes:
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        stack = [root]
        number_node = 0
        while stack:
            node = stack.pop()
            if node:
                number_node += 1
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return number_node

#223 Rectangle area
class Solution(object):
    def computeArea(self, A, B, C, D, E, F, G, H):
        """
        :type A: int
        :type B: int
        :type C: int
        :type D: int
        :type E: int
        :type F: int
        :type G: int
        :type H: int
        :rtype: int
        """

        left_corner_x = max(A, E)
        right_corner_x = min(C, G)
        left_corner_y = max(B, F)
        right_corner_y = min(D, H)
        area_1 = (C-A)*(D-B)
        area_2 = (G-E)*(H-F)
        if C<E or G<A or H<B or D<F:
            return area_1 + area_2
        return area_1+area_2-(right_corner_x - left_corner_x)*(right_corner_y - left_corner_y)


#218: The Skyline Problem:
from heapq import heappush, heappop
class Solution(object):
    def getSkyline(self, buildings):
        """
        :type buildings: List[List[int]]
        :rtype: List[List[int]]
        """
        critical_points =  [(L, -H, R) for L, R, H in buildings]
        critical_points += [(R, 0, 0) for _, R, _ in buildings]
        critical_points.sort()

        live_height = [(0, float('inf'))]
        res = [[0, 0]]

        for cur_pos, neg_h, cur_r in critical_points:
            while cur_pos >= live_height[0][1]:
                heappop(live_height)
            if neg_h:
                heappush(live_height, (neg_h, cur_r))
            if res[-1][1] != -live_height[0][0]:
                res.append([cur_pos, -live_height[0][0]])
        return res[1:]

# 225 implement stack using queues
class MyStack(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._data_1 = []
        self._data_2 = []
        self._top_num = None

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: None
        """
        self._data_1.append(x)
        self._top_num = x

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        if self._data_2:
            for _ in range(len(self._data_2)-1):
                self._top_num = self._data_2.pop(0)
                self._data_1.append(self._top_num)
            return self._data_2.pop(0)
        else:
            for _ in range(len(self._data_1)-1):
                self._top_num = self._data_1.pop(0)
                self._data_2.append(self._top_num)
            return self._data_1.pop(0)

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        return self._top_num

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return self._data_1 == [] and self._data_2 == []


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()

#224 basic calculator
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        res, num, sign, stack = 0, 0, 1, []
        for ss in s:
            if ss.isdigit():
                num = 10*num + int(ss)
            elif ss in ["-", "+"]:
                res += sign*num
                num = 0
                sign = -1 if ss == '-' else 1
            elif ss == "(":
                stack.append(res)
                stack.append(sign)
                sign, res = 1, 0
            elif ss == ")":
                res += sign*num
                res *= stack.pop()
                res += stack.pop()
                num = 0
        return res + num*sign

#226 Invert Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None
        left = root.left if root.left else None
        right = root.right if root.right else None

        self.invertTree(left)
        self.invertTree(right)
        root.left, root.right = right, left
        return root

# 227 Basic alculator II
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        pos = [0]
        sign = [1]
        stack = []
        for i, ch in enumerate(s):
            if ch in '+-':
                temp_sign = -1 if ch=='-' else 1
                sign.append(temp_sign)
                if pos[-1] < len(s):
                    stack.append(s[pos[-1]:i])
                pos.append(i+1)
        if pos[-1] < len(s):
            stack.append(s[pos[-1]:len(s)])
        # print(sign, stack)
        res = 0
        for sign_temp, exp in zip(sign, stack):
            #print(sign_temp, exp)
            res += sign_temp*(self.calculate_multiply_division_only(exp))
        return res
    def calculate_multiply_division_only(self, exp):
        stack = []
        num = 0
        for ch in exp:
            if ch.isdigit():
                num = num*10 + int(ch)
            elif ch in '*/':
                stack.append(num)
                stack.append(ch)
                num = 0
        stack.append(num)
        stack = stack[::-1]
        # print(stack)
        res = stack.pop()
        while stack:
            operation = stack.pop()
            num = stack.pop()
            if operation == '*':
                res = res * num
            elif operation == '/':
                res = res//num
        return res

# best solution:
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        pre, cur = 0, 0
        op = '+'
        num = 0
        for i, ch in enumerate(s):
            if ch.isdigit():
                num = num*10 + int(ch)
            if ch in '+-*/' or i==len(s)-1:
                if op == '+':
                    pre, cur = pre+cur, num
                elif op == '-':
                    pre, cur = pre+cur, -num
                elif op == '*':
                    cur *= num
                elif op == '/':
                    if cur<0:
                        cur = -(-cur//num)
                    else:
                        cur = cur//num
                op = ch
                num = 0
                # print(pre, cur, num)
        return pre+cur

#228: summary ranges:
class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        if not nums:
            return []
        res = []
        left = nums[0]
        right = None
        i = 1
        n = len(nums)
        cur = left
        while i<n:
            if nums[i] == cur+1:
                cur = nums[i]
            else:
                right=nums[i-1]
                if left == right:
                    res.append(str(left))
                else:
                    res.append(str(left) + '->' + str(right))
                left = nums[i]
                cur = left
                right = None
            i += 1
        if left == cur:
            res.append(str(left))
        else:
            res.append(str(left) + '->' + str(cur))
        return res

class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """

        start, end, i, result  = 0, 0, 1, []

        while i < len(nums) + 1 :
            if i != len(nums) and nums[end] + 1 == nums[i] :
                end = i
            else :
                if start == end :
                    s = str(nums[start])
                else :
                    s = str(nums[start]) + "->" + str(nums[end])

                start, end = i, i
                result.append(s)
            i += 1

        return result

# 229: Majority element II
class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """

        start, end, i, result  = 0, 0, 1, []

        while i < len(nums) + 1 :
            if i != len(nums) and nums[end] + 1 == nums[i] :
                end = i
            else :
                if start == end :
                    s = str(nums[start])
                else :
                    s = str(nums[start]) + "->" + str(nums[end])

                start, end = i, i
                result.append(s)
            i += 1

        return result
# 229 Majority Element II
# This is Boyer-Moore majority vote algorithm to solve it in O(1) space
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        candidate_1, candidate_2 = 0, 0
        count_1, count_2 = 0, 0
        for x in nums:
            if x == candidate_1:
                count_1 += 1
            elif x == candidate_2:
                count_2 += 1
            elif count_1 ==0:
                candidate_1 = x
                count_1 = 1
            elif count_2 == 0:
                candidate_2 = x
                count_2 = 1

            else:
                count_1 -= 1
                count_2 -= 1
        # second loof to confirm:
        # print(candidate_1, candidate_2)
        res = []
        count_1, count_2 = 0, 0
        for x in nums:
            if x == candidate_1:
                count_1 += 1
            if x == candidate_2:
                count_2 += 1
        if count_1 > len(nums)//3:
            res.append(candidate_1)

        if count_2 > len(nums) //3 and candidate_2!= candidate_1:
            res.append(candidate_2)
        return res
# solution for O(n) space:
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = {}
        for num in nums:
            if num in res:
                res[num] += 1
            else:
                res[num] = 1
        return [x for x in res if res[x] > (len(nums)//3)]
# 230 find kth smallest values in a BST
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        stack = [root]

        while stack[-1].left:
            stack.append(stack[-1].left)

        while stack:
            temp = stack.pop()
            k -=1
            if k ==0:
                return temp.val

            right = temp.right
            while right:
                stack.append(right)
                right = right.left
        return False

# Sol 2: InOrderTraversal
class Solution(object):

    def doInOrderTraversal(self, node):
        if not node or self.result:
            return
        self.doInOrderTraversal(node.left)
        self.nodeArray.append(node.val)
        if len(self.nodeArray) == self.limit:
            self.result = self.nodeArray[-1]
            return
        self.doInOrderTraversal(node.right)

    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        self.nodeArray = []
        self.result= None
        self.limit = k
        self.doInOrderTraversal(root)
        if not self.nodeArray:
            return None
        return self.result
# 231 Power of Two:
class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n<=0:
            return False
        if n==1:
            return True
        elif n % 2 == 1:
            return False
        elif int(n**0.5) == n**0.5:
            return self.isPowerOfTwo(n**0.5)
        elif int((n/2)**0.5) == (n/2)**0.5:
            return self.isPowerOfTwo((n/2)**0.5)
        else:
            return False

class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        return n>0 and n&(n-1) == 0

# Implement Queue using stacks:
class MyQueue(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._stack_1 = []
        self._stack_2 = []
        self._front_element = None
    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: None
        """
        if len(self._stack_1) == 0:
            self._front_element = x
        self._stack_1.append(x)


    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        if len(self._stack_2)==0:

            while self._stack_1:
                self._stack_2.append(self._stack_1.pop())
        return self._stack_2.pop()

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        if len(self._stack_2) != 0:
            return self._stack_2[-1]
        return self._front_element

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return len(self._stack_1) == 0 and len(self._stack_2) ==0


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# Palindrome linked list
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head:
            return True
        n = 0
        temp = head
        while temp:
            n += 1
            temp = temp.next
        second_half = head
        for _ in range(n//2-1):
            second_half = second_half.next
        temp = second_half.next
        second_half.next = None
        reverse = self.reverse_link_list(temp)
        #print(reverse.val)
        #print(head.val)
        if reverse is None:
            return True
        for _ in range(n//2):
            if head.val != reverse.val:
                return False
            head = head.next
            reverse = reverse.next
        return True

    def reverse_link_list(self, head):
        if not head:
            return None
        reverse, cur = None, head
        while cur.next:
            temp = cur.next
            cur.next = reverse
            reverse = cur
            cur = temp
        cur.next = reverse
        return cur

def isPalindrome(self, head):
        def reverse(node):
            prev = None
            curr = node
            next = None
            while curr:
                next = curr.next
                curr.next = prev
                prev = curr
                curr = next
            return prev

        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        mid_rev = reverse(slow)
        start = head
        while mid_rev:
            if start.val != mid_rev.val:
                return False
            start = start.next
            mid_rev = mid_rev.next

        return True

# 235 Lowest common ancestor of a binary search Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        parent_val = root.val
        p_val = p.val
        q_val = q.val
        if p_val > parent_val and q_val > parent_val:
            return self.lowestCommonAncestor(root.right, p, q)
        elif p_val<parent_val and q_val < parent_val:
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root

# 233 Number of digit one
import math
class Solution(object):
    def countDigitOne(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n<1:
            return 0
        if n==1:
            return 1

        res = 0

        n_digits = int(math.log10(n)) + 1

        for i in range(n_digits):
            base = 10**(i+1)
            # number of 1 at the position i (from the right). for example i=1,
            # it is number of 1 at the second position from the right (i.e. 1x)
            res += (n//(base))*(base//10) + self.count_1_per_base(n%base, base//10)
        return res

    def count_1_per_base(self, n, base):
        if base == 1:
            return int(n>=base)
        res  = 0
        if n//base>1:
            res += base
        elif n//base==1:
            res += n%base +1
        return res

#236: Lowest Common Ancestor of a Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        self.ans = None
        def node_search(node, p, q):
            if not node:
                return False

            left = node_search(node.left, p, q)

            right = node_search(node.right, p, q)

            if node == p or node ==q:
                mid = True
            else:
                mid = False
            if left+right+mid>=2:
                self.ans = node
            return mid or left or right

        node_search(root, p, q)
        return self.ans

# solution 2: using stack
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        parent = {root: None}
        stack = [root]
        while stack:
            node = stack.pop()
            if node.left:
                parent[node.left] = node
                stack.append(node.left)
            if node.right:
                parent[node.right] = node
                stack.append(node.right)

        # Ancestors set for node p
        ancestors = set()
        while p:
            ancestors.add(p)
            p = parent[p]
        # interact back q from their ancestors util it is in ancestors
        while q not in ancestors and q:
            q = parent[q]
        return q
# Sol 3: complicated
class Solution:

    # Three static flags to keep track of post-order traversal.

    # Both left and right traversal pending for a node.
    # Indicates the nodes children are yet to be traversed.
    BOTH_PENDING = 2
    # Left traversal done.
    LEFT_DONE = 1
    # Both left and right traversal done for a node.
    # Indicates the node can be popped off the stack.
    BOTH_DONE = 0

    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """

        # Initialize the stack with the root node.
        stack = [(root, Solution.BOTH_PENDING)]

        # This flag is set when either one of p or q is found.
        one_node_found = False

        # This is used to keep track of LCA index.
        LCA_index = -1

        # We do a post order traversal of the binary tree using stack
        while stack:

            parent_node, parent_state = stack[-1]

            # If the parent_state is not equal to BOTH_DONE,
            # this means the parent_node can't be popped of yet.
            if parent_state != Solution.BOTH_DONE:

                # If both child traversals are pending
                if parent_state == Solution.BOTH_PENDING:

                    # Check if the current parent_node is either p or q.
                    if parent_node == p or parent_node == q:

                        # If one_node_found is set already, this means we have found both the nodes.
                        if one_node_found:
                            return stack[LCA_index][0]
                        else:
                            # Otherwise, set one_node_found to True,
                            # to mark one of p and q is found.
                            one_node_found = True

                            # Save the current top index of stack as the LCA_index.
                            LCA_index = len(stack) - 1

                    # If both pending, traverse the left child first
                    child_node = parent_node.left
                else:
                    # traverse right child
                    child_node = parent_node.right

                # Update the node state at the top of the stack
                # Since we have visited one more child.
                stack.pop()
                stack.append((parent_node, parent_state - 1))

                # Add the child node to the stack for traversal.
                if child_node:
                    stack.append((child_node, Solution.BOTH_PENDING))
            else:

                # If the parent_state of the node is both done,
                # the top node could be popped off the stack.

                # i.e. If LCA_index is equal to length of stack. Then we decrease LCA_index by 1.
                if one_node_found and LCA_index == len(stack) - 1:
                    LCA_index -= 1
                stack.pop()

        return None
# 237 Delete Node in a linked list:
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next

# 238 product of array except self
class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        product = 1
        count_zeros = 0
        for num in nums:
            if num !=0:
                product *= num
            else:
                count_zeros += 1
        output = []
        for i, num in enumerate(nums):
            if count_zeros == 0:
                output.append(product//num)
            elif count_zeros >1:
                output.append(0)
            elif num==0:
                output.append(product)
            else:
                output.append(0)

        return output
# 242 Valid anagram:
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        s_list = sorted([x for x in s])
        t_list = sorted([x for x in t])
        return s_list == t_list

# best solution
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) != len(t):
            return False
        counter = [0] * 26
        for i in range(len(s)):
            counter[ord(s[i]) - ord('a')] += 1
            counter[ord(t[i]) - ord('a')] -= 1
        return all([x == 0 for x in counter])

# 240 Search a 2D matrix II
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if matrix and matrix[0]:
            num_row = len(matrix)
            num_col = len(matrix[0])
            row = num_row - 1
            col = 0
            while num_row > row>= 0 and 0<=col<num_col:
                if matrix[row][col] > target:
                    row -= 1
                elif matrix[row][col] < target:
                    col += 1
                else:
                    return True
        return False

# 241 Different ways to add parentheses
from operator import *
class Solution(object):
    def diffWaysToCompute(self, input):
        """
        :type input: str
        :rtype: List[int]
        """
        ops = {"+": add, "-": sub, "*": mul, "/": div}
        ans = []
        for i, c in enumerate(input):
            if c in ops:
                left = self.diffWaysToCompute(input[:i])
                right = self.diffWaysToCompute(input[i+1:])
                ans.extend([ops[c](a, b) for a in left for b in right])
        return ans if ans else [int(input)]

# 257 Binary Tree Paths
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        if not root:
            return []
        if not root.left and not root.right:
            return [str(root.val)]
        ans, left, right = [], [], []
        if root.left:
            left = self.binaryTreePaths(root.left)
            ans += [str(root.val) + '->' + x for x in left]
        if root.right:
            right = self.binaryTreePaths(root.right)
            ans += [str(root.val) + '->' + x for x in right]
        return ans

# 258 Add digits:
class Solution(object):
    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        if num == 0:
            return 0
        return 1+ (num-1) % 9
class Solution(object):
    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        t = list(str(num))
        while len(t) > 1:
            num = sum(map(int, t))
            t = list(str(num))
        return num

# 239 Slidign Window maximum:
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        q = collections.deque()
        out = []
        for i, num in enumerate(nums):
            while q and nums[q[-1]] < num:
                q.pop()
            q.append(i)
            if q[0]==i-k:
                q.popleft()
            if i>=k-1:
                out.append(nums[q[0]])
        return out
# sol 2: using heap
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        counter, heap, out = collections.Counter(), [], []
        for i, num in enumerate(nums):
            counter[num] += 1
            heapq.heappush(heap, -num)
            while not counter[-heap[0]]:
                heapq.heappop(heap)
            if i>=k-1:
                out.append(-heap[0])
                counter[nums[i-k+1]] -=1
        return out
#260 Single Number III
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        counter = collections.Counter(nums)
        return [x for x in counter if counter[x]==1]

# 262 Trips and Users
# Write your MySQL query statement below
SELECT
Request_at as Day,
ROUND( SUM(case when substring(Status, 1,  9)="cancelled" then 1 else 0 end)/count(Status) ,2 ) as "Cancellation Rate"
FROM Trips
WHERE Client_Id in
(select Users_Id from Users where Banned="No") and
Request_at BETWEEN "2013-10-01" and "2013-10-03"
GROUP BY Request_at

# 263 Ugly number
class Solution(object):
    def isUgly(self, num):
        """
        :type num: int
        :rtype: bool
        """
        if num<1:
            return False
        if num==1 or num==2 or num==3 or num==5:
            return True
        if num/2 == num/2.0:
            return self.isUgly(num//2)
        elif num/3 == num/3.0:
            return self.isUgly(num//3)
        elif num/5 == num/5.0:
            return self.isUgly(num//5)
        else:
            return False

# 264 Ugly Number II:
class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        ugly=[1]
        i2, i3, i5=0, 0, 0
        while n>1:
            u_min = min(2*ugly[i2], 3*ugly[i3], 5*ugly[i5])
            if u_min == 2*ugly[i2]:
                i2 += 1
            if u_min == 3*ugly[i3]:
                i3 += 1
            if u_min == 5*ugly[i5]:
                i5 += 1
            ugly.append(u_min)
            n -= 1
        return ugly[-1]

# Solution 2 based on XOR (^).
"""
Well, the XOR solution is based on the fact that the largest element
in the list is the one that replaced the missing element we are
searching for. This means the largest element can never be missing.
The actual list contains integers from 0 to n, the list we are given
has 1 missing element.
So we can conclude n = length of the given list
What you need to know about XOR is that it's both commutive and
assosiative
commutive means A ^ B = B ^ A
assosiative means (A ^ B) ^ C = A ^ (B ^ C)
This means that order doesn't matter
Other thing you need to know about XOR, if you XOR 2 similar numbers,
the result is 0. A ^A = 0
So now we will XOR the value of n with all the indicies from 0 to n-1
and with all the values we have in our array.
For example:
array = 0,1,2,4
index = 0,1,2,3
n = 4
4 ^ (0^0) ^ (11) ^ (2^2) ^ (4^3)
= (4^4) ^ (0^0) ^ (11) ^ (2^2) ^ 3
= 0 ^ 0 ^ 0 ^ 0 ^ 3
= 3
"""
class Solution:
    def missingNumber(self, nums):
        missing = len(nums)
        for i, num in enumerate(nums):
            missing ^= i ^ num
        return missing
# 273 H index:
class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        if not citations:
            return 0
        #if len(citations) == 1:
        #    return int(citations[0] >= 1)
        citations.sort()
        citations = citations[::-1]
        n = len(citations)
        # print(citations)
        for i in range(n):
            if i<n-1 and citations[i] >= i+1 and citations[i+1] <= i+1:
                return i+1
            elif i==n-1 and citations[i] >= i+1:
                return i+1
        return 0

class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """

        citations.sort(reverse=True)

        for i in range(len(citations)):
            if citations[i] < i+1:
                return i

        return len(citations)

class Solution:
    def hIndex(self, citations: List[int]) -> int:
        citations = sorted(citations, reverse=True)
        count_citations = 0
        h = len(citations)
        i = 0
        while(i < len(citations)):
            citation = citations[i]
            if citation >= h:
                count_citations += 1
                i += 1
            else:
                h -= 1
            if h == count_citations:
                return h
        return 0

#274 Integer to english words

class Solution(object):
    def numberToWords(self, num):
        to19 = 'One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve ' \
               'Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen'.split()
        tens = 'Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety'.split()
        def words(n):
            if n < 20:
                return to19[n-1:n]
            if n < 100:
                return [tens[n/10-2]] + words(n%10)
            if n < 1000:
                return [to19[n/100-1]] + ['Hundred'] + words(n%100)
            for p, w in enumerate(('Thousand', 'Million', 'Billion'), 1):
                if n < 1000**(p+1):
                    return words(n/1000**p) + [w] + words(n%1000**p)
        return ' '.join(words(num)) or 'Zero'

#275 H-Index II
class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        if not citations:
            return 0
        n = len(citations)
        left, right = 0, n
        while left<right:

            mid = (left+right)//2
            # print(n-mid, citations[mid])
            if citations[mid] > n-mid:
                right = mid
            elif citations[mid] <n-mid:
                left = mid+1
            else:
                return n-mid
        return n-left

# 278 First bad version
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        left = 1
        right = n
        while left<right:
            mid = (left + right)//2
            if isBadVersion(mid):
                right = mid
            else:
                left = mid+1
        return left

# 279 Perfect Squares
lass Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """

        if n<=1:
            return 1
        dp = [None]*(n+1)
        dp[0], dp[1] = 0, 1

        for i in range(2, n+1):
            q = 1
            min_square = float('inf')
            while q*q<=i:
                # print(q, self.numSquares(n-q*q))
                min_square = min(min_square, 1 + dp[i-q*q])
                q += 1
            dp[i] = min_square
        return dp[n]

# 282 Expression Add Operators
class Solution:
    def addOperators(self, num, target):

        N = len(num)
        answers = []
        def recurse(index, prev_operand, current_operand, value, string):

            # Done processing all the digits in num
            if index == N:

                # If the final value == target expected AND
                # no operand is left unprocessed
                if value == target and current_operand == 0:
                    answers.append("".join(string[1:]))
                return

            # Extending the current operand by one digit
            current_operand = current_operand*10 + int(num[index])
            str_op = str(current_operand)

            # To avoid cases where we have 1 + 05 or 1 * 05 since 05 won't be a
            # valid operand. Hence this check
            if current_operand > 0:

                # NO OP recursion
                recurse(index + 1, prev_operand, current_operand, value, string)

            # ADDITION
            string.append('+'); string.append(str_op)
            recurse(index + 1, current_operand, 0, value + current_operand, string)
            string.pop();string.pop()

            # Can subtract or multiply only if there are some previous operands
            if string:

                # SUBTRACTION
                string.append('-'); string.append(str_op)
                recurse(index + 1, -current_operand, 0, value - current_operand, string)
                string.pop();string.pop()

                # MULTIPLICATION
                string.append('*'); string.append(str_op)
                recurse(index + 1, current_operand * prev_operand, 0, value - prev_operand + (current_operand * prev_operand), string)
                string.pop();string.pop()
        recurse(0, 0, 0, 0, [])
        return answers

#283 Move Zeroes
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        if len(nums)<=1:
            return
        pointer_0, pointer_non_zero = 0, 0

        while pointer_non_zero < len(nums):
            while pointer_non_zero <len(nums) and nums[pointer_non_zero] == 0:
                pointer_non_zero +=1

            while pointer_0<len(nums) and nums[pointer_0] != 0:
                pointer_0 += 1
            # print(pointer_0, pointer_non_zero)
            if pointer_non_zero<len(nums) and pointer_0 < len(nums) and pointer_non_zero>pointer_0:
                nums[pointer_0], nums[pointer_non_zero] = nums[pointer_non_zero], nums[pointer_0]
            pointer_non_zero = pointer_non_zero + 1
        return

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        #count the zeros
        num_zeroes = 0
        for n in nums:
            if n == 0:
                num_zeroes += 1
        #Make all the non-zero elements retain their original order
        res = [n for n in nums if n != 0]

        #Move all zeroes to the end
        res += [0] * num_zeroes
        nums = res
        #return nums

class Solution2:
    def moveZeroes(self, nums: List[int]) -> None:
        slow = 0
        ##if the current element is not 0, then we need to
        ##append it just ijn front of last non 0 element we found
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[slow] = nums[i]
                slow += 1
        #fulfill remaining array with 0's
        for i in range(slow, len(nums)):
            nums[i] = 0

        #return nums

class Solution3:
   def moveZeroes(self, nums: List[int]) -> None:
        slow = cur = 0
        for cur in range(len(nums)):
            if nums[cur] != 0:
                #swap the postion of slow pointer and cur
                nums[slow], nums[cur] = nums[cur], nums[slow]
                slow += 1
        #return nums

# 284 Peeking Iterator
# Below is the interface for Iterator, which is already defined for you.
#
# class Iterator(object):
#     def __init__(self, nums):
#         """
#         Initializes an iterator object to the beginning of a list.
#         :type nums: List[int]
#         """
#
#     def hasNext(self):
#         """
#         Returns true if the iteration has more elements.
#         :rtype: bool
#         """
#
#     def next(self):
#         """
#         Returns the next element in the iteration.
#         :rtype: int
#         """

class PeekingIterator(object):
    def __init__(self, iterator):
        self.iter = iterator
        self.temp = self.iter.next() if self.iter.hasNext() else None

    def peek(self):
        return self.temp

    def next(self):
        ret = self.temp
        self.temp = self.iter.next() if self.iter.hasNext() else None
        return ret

    def hasNext(self):
        return self.temp is not None


# Your PeekingIterator object will be instantiated and called as such:
# iter = PeekingIterator(Iterator(nums))
# while iter.hasNext():
#     val = iter.peek()   # Get the next element but not advance the iterator.
#     iter.next()         # Should return the same value as [val].

#282 Expression add operators
class Solution(object):
    def addOperators(self, num, target):
        """
        :type num: str
        :type target: int
        :rtype: List[str]
        """
        N = len(num)
        ans = []

        def recurse(index, prev_operand, current_operand, value, string):

            # Done processing all the digits in num
            if index==N:
                # if the final value = target expected and
                # no operand is left unprocessed
                if value == target and current_operand == 0:
                    ans.append("".join(string[1:]))
                return
            # Extending the current operand by one digit
            current_operand = current_operand*10 + int(num[index])
            str_op = str(current_operand)
            # To avoid cases where we have 1+05 or 1*05 since 05 is not a
            # valid operand. we need to perform current_operand
            if current_operand>0:
                # No operation recursion
                recurse(index+1, prev_operand, current_operand, value, string)
            # Addition
            string.append('+')
            string.append(str_op)
            recurse(index+1, current_operand, 0, value + current_operand, string)
            string.pop()
            string.pop()

            # can suctract or multiply only if there are some previous operands
            if string:
                # Subtraction
                string.append('-')
                string.append(str_op)
                recurse(index+1, -current_operand, 0, value-current_operand, string)
                string.pop()
                string.pop()

                # Multiplication
                string.append('*')
                string.append(str_op)
                recurse(index+1, current_operand*prev_operand, 0, value-prev_operand+(current_operand*prev_operand), string)
                string.pop()
                string.pop()
        recurse(0, 0, 0, 0, [])
        return ans

# 287 Find the duplicate number:
# this is similar to cycle link list II with a link is index->nums[index]
class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        fast = nums[0]
        slow = nums[0]
        while True:
            fast = nums[nums[fast]]
            slow = nums[slow]
            if slow == fast:
                break
        pt1 = nums[0]
        pt2 = slow
        while pt1!=pt2:
            pt1=nums[pt1]
            pt2=nums[pt2]
        return pt1

# 289 Game of Life:
class Solution(object):
    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        max_row = len(board)
        max_col = len(board[0])
        new_board = [[None]*max_col for _ in range(max_row)]
        for r in range(max_row):
            for c in range(max_col):
                new_board[r][c] = self.next_live_dead(board, r, c)
        for r in range(max_row):
            for c in range(max_col):
                board[r][c] = new_board[r][c]
        return board
    def next_live_dead(self, board, row, col):
        max_row = len(board)
        max_col = len(board[0])
        neigh_row = [row-1, row, row+1]
        neigh_col = [col-1, col, col+1]
        count_1 = 0
        for r in neigh_row:
            for c in neigh_col:
                if r==row and c==col:
                    cel_val = board[row][col]
                elif 0 <= r < max_row and 0 <= c < max_col and board[r][c]==1:
                    count_1 +=1

        if cel_val == 1 and count_1<2:
            return 0
        elif cel_val == 1 and (count_1==2 or count_1==3):
            return 1
        elif cel_val == 1 and count_1 >3:
            return 0
        elif cel_val == 0 and count_1==3:
            return 1
        else:
            return 0

# Sol 2:
"""
As the desscription:
You cannot update some cells first and then use their updated values to update other cells.

If we wanted to change the state in place,
We need to set another number that can represent the state now and the next state
So we don't mess up our count.

So I use
3 means alive now and going to die,
2 means dead now and going alive.

All the '%2' in the count_alive() is just a way to make it cleaner [0]
Avoiding bunch of if-else
Because 3%2==1 and 2%2==0

Time Complexity is O(MN)
Because we go through the whole board once [1]
Each time at a (i,j) we use O(1) to count its alive neighbor [2]
And go through the whole board another time to convert 3->0 and 2->1 [3]
O(MN*1+MN)~=O(MN)

Space Complexity is O(1)
Because we done it in place.

I learn this really nice and clean solution from @zhuyinghua1203
"""
class Solution(object):
    def gameOfLife(self, board):
        if board==None or len(board)==0 or len(board[0])==0: return board

        m = len(board)
        n = len(board[0])

        def count_alive(i, j):
            if i<0 or j<0 or i>=m or j>=n: return 0
            count = 0

            #bottom, top, right, left
            if i+1<m: count+=board[i+1][j]%2 #[0]
            if 0<=i-1: count+=board[i-1][j]%2
            if j+1<n: count+=board[i][j+1]%2
            if 0<=j-1: count+=board[i][j-1]%2

            #bottomright, topleft, bottomleft, topright
            if i+1<m and j+1<n: count+=board[i+1][j+1]%2
            if 0<=i-1 and 0<=j-1: count+=board[i-1][j-1]%2
            if i+1<m and 0<=j-1: count+=board[i+1][j-1]%2
            if 0<=i-1 and j+1<n: count+=board[i-1][j+1]%2

            return count

        for i in xrange(m): #[1]
            for j in xrange(n):
                count = count_alive(i, j) #[2]

                if board[i][j]==1:
                    if count<2 or count>3:
                        board[i][j] = 3
                elif board[i][j]==0:
                    if count==3:
                        board[i][j] = 2

        for i in xrange(m): #[3]
            for j in xrange(n):
                if board[i][j]==2:
                    board[i][j] = 1
                elif board[i][j]==3:
                    board[i][j] = 0

        return board

# 290 Word Pattern
class Solution(object):
    def wordPattern(self, pattern, str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        word_dict = {}
        str_list = str.split()
        if len(str_list) != len(pattern):
            return False
        for ch, word in zip(pattern, str_list):
            if word not in word_dict.values():
                word_dict[ch] = word
            elif ch not in word_dict:
                return False

        new_str = " ".join([word_dict[ch] for ch in pattern])
        return new_str == str

class Solution(object):
    def wordPattern(self, pattern, str_):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        d_hash = {}
        str_ = str_.split()

        if len(pattern) != len(str_):
            return False

        for p, s in zip(pattern, str_):
            if p in d_hash:
                if d_hash[p] != s:
                    return False
            elif s in d_hash.values():
                return False
            else:
                d_hash[p] = s
        return True

# 292 Nim Game:
class Solution(object):
    def canWinNim(self, n):
        """
        :type n: int
        :rtype: bool
        """
        return n%4>0
# 295 Find median from data stream:
from heapq import *
class MedianFinder(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self._low = []
        self._high = []


    def addNum(self, num):
        """
        :type num: int
        :rtype: None
        """
        heappush(self._low, -num)
        heappush(self._high, -heappop(self._low))
        if len(self._high) > len(self._low):
            heappush(self._low, -heappop(self._high))

    def findMedian(self):
        """
        :rtype: float
        """
        if len(self._high)==len(self._low):
            return (-self._low[0] + self._high[0])/2.0
        return -self._low[0]



# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()

# 297 Serialize and deserialize Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """

        if root == None: return None
        res = {"val": root.val,
               "left": self.serialize(root.left) if root.left else None,
               "right": self.serialize(root.right) if root.right else None}
        return res


    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        if data == None: return None
        root = TreeNode(data["val"])
        root.left = self.deserialize(data["left"])
        root.right = self.deserialize(data["right"])
        return root


# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))

# 299 Bulls and cows
class Solution(object):
    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """
        bull = 0
        cow = 0
        secret_list, guess_list = [], []
        for s, g in zip(secret, guess):
            if s==g:
                bull += 1
            else:
                secret_list.append(s)
                guess_list.append(g)
        for ch in guess_list:
            if ch in secret_list:
                cow +=1
                secret_list.remove(ch)
        return str(bull) + 'A' + str(cow) + 'B'

# 300 Longest increasing subsequence
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        dp=[1] * len(nums)
        for i in range(len(nums)):
            max_length = 0
            for j in range(i):
                if nums[j] < nums[i]:
                    max_length = max(max_length, dp[j])
            dp[i] = max_length + 1
        return max(dp)

# Python program to find
# length of longest
# increasing subsequence
# in O(n Log n) time

# Binary search (note
# boundaries in the caller)
# A[] is ceilIndex
# in the caller
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def binary_search(nums, l, r, key):
            # search to find the index of key
            # if we need to insert key into
            # the order array nums
            while r-l>1:
                m = l + (r-l)//2
                if nums[m] >= key:
                    r = m
                else:
                    l = m
            return r
        size = len(nums)
        tail_table = [0 for _ in range(size + 1)]
        len_LIS = 0 # always points empty slot
        tail_table[0] = nums[0]
        len_LIS = 1
        for i in range(1, size):
            if nums[i] < tail_table[0]:
                # new smallest value
                tail_table[0] = nums[i]
            elif nums[i] > tail_table[len_LIS-1]:
                # num[i] wants to extend largest subsequence
                tail_table[len_LIS] = nums[i]
                len_LIS += 1
            else:
                # nums[i] wants to be current
                # end candidate of an existing
                # subsequence. It will replace
                # ceil value in tail_table
                tail_table[binary_search(tail_table, -1, len_LIS-1, nums[i])] = nums[i]
        return len_LIS

# 303 Range sum query - Immutable
class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self._nums = nums
        self._length = len(nums)
        self._sum = [0] * (len(nums)+1)
        for i in range(0, len(nums)):
            self._sum[i+1] = self._sum[i] + nums[i]

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        if i>=self._length or j>=self._length or i<0 or j<0 or i>j:
            return None

        return self._sum[j+1] - self._sum[i]


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(i,j)

# 301 Remove Invalid Parentheses
class Solution(object):
    def removeInvalidParentheses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        misplaced_left, misplaced_right = 0, 0
        for ch in s:
            if ch == '(':
                misplaced_left += 1
            elif ch == ')':
                misplaced_right += 1
                if misplaced_left >0:
                    misplaced_left -= 1
                    misplaced_right -=1

        def recursion(index, left, right, left_rem, right_rem, cur_exp):
            if index==len(s):
                if left == right and left_rem == right_rem == 0:
                    self.ans.add(cur_exp)
                    return
            else: # s[index] can only be '('; ')' or a letter
                if s[index] == '(':
                    if left_rem > 0: # we can remove left parentheses
                        recursion(index+1, left, right, left_rem-1, right_rem, cur_exp)
                    recursion(index+1, left+1, right, left_rem, right_rem, cur_exp+'(') # keep current left parentheses
                elif s[index] == ')':
                    if right_rem>0: # if we can remove right parenthesis
                        recursion(index+1, left, right, left_rem, right_rem-1, cur_exp)
                    if left>right: # if we can keep the right parenthesis, we can only keep if right<left
                        recursion(index+1, left, right+1, left_rem, right_rem, cur_exp + ')')
                else:
                    recursion(index+1, left, right, left_rem, right_rem, cur_exp + s[index])
        self.ans = set()
        recursion(0, 0, 0, misplaced_left, misplaced_right, '')

        return list(self.ans)

# 304 Range Sum Query 2D-Immutable
class NumMatrix(object):

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        self._matrix = matrix
        if matrix:
            n_row = len(matrix)
            n_col = len(matrix[0])
        else:
            n_row, n_col = 0, 0
        self._sum = [[0] * (n_col+1) for _ in range(n_row+1)]

        for i_row in range(n_row):
            for i_col in range(n_col):
                self._sum[i_row+1][i_col+1] = self._sum[i_row+1][i_col] + matrix[i_row][i_col] + self._sum[i_row][i_col+1] - self._sum[i_row][i_col]
        print(self._sum)

    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        return self._sum[row2+1][col2+1] - self._sum[row1][col2+1] - self._sum[row2+1][col1] + self._sum[row1][col1]


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)
# 306 Additive Number
class Solution(object):
    def isAdditiveNumber(self, num):
        """
        :type num: str
        :rtype: bool
        """
        for i in range(1, len(num) - 1):
            for j in range(i + 1, len(num)):
                s1, s2 = num[:i], num[i: j]

                if (len(s1) > 1 and s1[0] == '0') or (len(s2) > 1 and s2[0] == '0'):
                    continue

                curr = s1 + s2
                while len(curr) < len(num):
                    s3 = str(int(s1) + int(s2))
                    curr += s3
                    s1, s2 = s2, s3

                if curr == num:
                    return True

        return False

# 30
# Binary Indexed Tree
class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        n = len(nums)

        self.nums = nums
        self.bit = [0] * (n + 1)

        for i in range(n):
            self._update(i + 1, nums[i])

    def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: None
        """
        self._update(i + 1, val - self.nums[i])
        self.nums[i] = val

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self._sum(j + 1) - self._sum(i)

    def _update(self, i, val):
        while i < len(self.bit):
            self.bit[i] += val
            i += i & -i

    def _sum(self, i):
        total = 0
        while i > 0:
            total += self.bit[i]
            i -= i & -i
        return total

# 309 Best time to buy and sell stock with cooldown
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        rest, buy, sell = [None]*len(prices), [None]*len(prices), [None]*len(prices)
        rest[0] = 0
        sell[0] = 0
        buy[0] = -prices[0]
        for i in range(1, len(prices)):
            rest[i] = max(rest[i-1], sell[i-1]) # you either rest at i-1 or sell at i-1
            sell[i] = buy[i-1]+prices[i] # you either sell at i-1 and rest or buy at i-1 and sell
            buy[i] = max(buy[i-1], rest[i-1]-prices[i]) # you either buy at i-1 and rest or rest at i-1 and buy
        return max(rest[-1], sell[-1], buy[-1])

#O(1) space solution
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        rest, buy, sell = 0, -prices[0], 0

        for i in range(1, len(prices)):
            temp_rest, temp_sell, temp_buy = rest, sell, buy
            rest = max(temp_rest, temp_sell)  # you either rest at i-1 or sell at i-1
            sell = temp_buy+prices[i] # you either sell at i-1 and rest or buy at i-1 and sell
            buy = max(temp_buy, temp_rest-prices[i]) # you either buy at i-1 and rest or rest at i-1 and buy

        return max(rest, sell, buy)
# 310 Minimum Height Trees
# BFS
class Solution(object):
    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        if not n: return []
        connect = [[] for _ in range(n)]
        deg = [0] * n
        for u, v in edges:
            connect[u].append(v)
            connect[v].append(u)
            deg[u] += 1
            deg[v] += 1
        leaves = [x for x in range(n) if deg[x]<=1]
        done = set(leaves)
        while len(done) < n:
            next = []
            for x in leaves:
                for u in connect[x]:
                    if u in done:
                        continue
                    deg[u] -=1
                    if deg[u] == 1:
                        done.add(u)
                        next.append(u)
            leaves = next
            # print(leaves)
        return list(leaves)

class Solution:
    def findMinHeightTrees(self, n, edges):
        if not n: return []
        e = [[] for _ in range(n)]
        for u, v in edges:
            e[u].append(v)
            e[v].append(u)

        def longest_path(u, par=None):
            mx = []
            for v in e[u]:
                if v != par:
                    now = longest_path(v, u)
                    if len(now) > len(mx):
                        mx = now
            mx.append(u)
            return mx
        fr = longest_path(0)[0]
        path = longest_path(fr)
        k = len(path)
        return path[(k-1)>>1 : (k+2)>>1]

def findMinHeightTrees(n, edges):
    tree = [set() for _ in range(n)]
    for u, v in edges: tree[u].add(v), tree[v].add(u)
    q, nq = [x for x in range(n) if len(tree[x]) < 2], []
    while True:
        for x in q:
            for y in tree[x]:
                tree[y].remove(x)
                if len(tree[y]) == 1: nq.append(y)
        if not nq: break
        nq, q = [], nq
    return q

# 312 Burst ballons
class Solution(object):
    def maxCoins(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        memo = {}
        nums = [1] + nums + [1]
        def dfs(l, r):
            # maximum coin to get when burting all balloons
            # from left+1 to right-1
            if r-l==1: return 0
            max_coin = 0
            if (l, r) not in memo:
                for k in range(l+1, r):
                    max_coin = max(max_coin, dfs(l, k)+nums[l]*nums[k]*nums[r]+dfs(k, r))
                memo[(l,r)] = max_coin
            # print(memo)
            return memo[(l,r)]
        return dfs(0, len(nums)-1)

class Solution(object):
    def maxCoins(self, nums):
        if not nums:
            return 0
        N = len(nums)
        nums = [1] + nums + [1]
        dp = [[0 for i in range(N+2)] for j in range(N+2)]
        for l in range(1, N+1):
            for start in range(1, N+1):
                end = start + l - 1
                if end > N:
                    break
                if l == 1:
                    dp[start][end] = nums[start-1]*nums[start]*nums[start+1]
                else:
                    max_ = 0
                    for last in range(start, end+1):
                        v = dp[start][last-1] + dp[last+1][end] + nums[start-1]*nums[last]*nums[end+1]
                        if v > max_:
                            max_ = v
                    dp[start][end] = max_
        return dp[1][N]

# Super Ugly Number:
class Solution(object):
    def nthSuperUglyNumber(self, n, primes):
        """
        :type n: int
        :type primes: List[int]
        :rtype: int
        """
        k = len(primes)
        index = [0] * k
        ugly = [1]
        while n>1:
            temp = [ugly[index[i]]*primes[i] for i in range(k)]
            u_min = min(temp)
            idx_min = temp.index(u_min)
            index[idx_min] += 1
            if u_min>ugly[-1]:
                ugly.append(u_min)
                n -= 1
        # print(ugly)
        return ugly[-1]
# An improvement:
class Solution:
    def nthSuperUglyNumber(self, N,primes):
        ugly_numbers = [0] * N
        ugly_numbers[0] = 1
        indexes = [0]* len(primes)
        refactors = [i for i in primes]
        for i in range(1, N):
            ugly_numbers[i] = min(refactors)
            for ind in [k for k, x in enumerate(refactors) if x == ugly_numbers[i]]:
                indexes[ind] += 1
                refactors[ind] = ugly_numbers[indexes[ind]]*primes[ind]
        return ugly_numbers[-1]

class Solution:
    def nthSuperUglyNumber(self, n, primes):
        """
        :type n: int
        :type primes: List[int]
        :rtype: int
        """
        h, heap = set([1]), [1]
        while n:
            a = heappop(heap)
            for i in primes:
                m = a * i
                if not m in h:
                    heappush(heap, m)
                    h.add(m)
            n -= 1
        return a

class Solution:
    def nthSuperUglyNumber(self, n, P):
        res, heap = [1], [(P[i], P[i], 0) for i in range(len(P))]
        while len(res) < n:
            val, prm, idx = heapq.heappop(heap)
            if val <= res[-1]:
                while val <= res[-1]: idx += 1; val = prm * res[idx]
            else:
                res += val,
                val, idx = prm * res[idx + 1], idx + 1
            heapq.heappush(heap, (val, prm, idx))
        return res[-1]

# 368 Largest divisible Subset
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
class Node(object):
        def __init__(self,val,parent):
            self.val=val
            self.children=[]
            self.depth=0
            self.parent=parent
        def append(self,val):
            found=False
            for child in self.children:
                if(val%child.val==0):
                    child.append(val)
                    found=True
            if(not found):
                self.children.append(Node(val,self))
                self.update_depth(1)
        def get_longest_path(self,path):
            path.append(self.val)
            if(len(self.children)==0):
                return path
            maximum_depth=-1
            max_index=-1
            for i in range(0,len(self.children)):
                child_depth=self.children[i].depth
                if(child_depth>maximum_depth):
                    maximum_depth=child_depth
                    max_index=i
            self.children[max_index].get_longest_path(path)
            return path
        def update_depth(self,depth):
            if(depth<self.depth):
                return
            else:
                self.depth=depth
                if(self.parent is not None):
                    self.parent.update_depth(depth+1)
            return

class Solution(object):
    def largestDivisibleSubset(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if(nums is None):
            return []
        n=len(nums)
        if(n<=1):
            return nums
        nums.sort()
        res=[]
        if(nums[0]==0): #zero could be part of any solution
            res.append(0)
            del nums[0]
        if(nums[0]==1): #to build the tree we need a root node with value 1, here we take care of the fact if it is in the input or not
            res.append(1)
            del nums[0]
        root=Node(1,None)
        for num in nums:
            root.append(num)
        longest_path=root.get_longest_path([])
        del longest_path[0] #delete the root node of 1 at the start of the path
        res+=longest_path
        return res

# 354 Russian Doll Envelopes
"""
You have a number of envelopes with widths and heights given as a pair
of integers (w, h). One envelope can fit into another if and only if
both the width and height of one envelope is greater than the width and
height of the other envelope.

What is the maximum number of envelopes can you Russian doll? (put one
inside other)

Note:
Rotation is not allowed.

Example:

Input: [[5,4],[6,4],[6,7],[2,3]]
Output: 3
Explanation: The maximum number of envelopes you can Russian doll is 3
([2,3] => [5,4] => [6,7]).
"""


class Solution(object):
    def maxEnvelopes(self, envelopes):
        """
        :type envelopes: List[List[int]]
        :rtype: int
        """
        if not envelopes:
            return 0
        envelopes = sorted(envelopes, key=lambda x: (x[0], -x[1]))
        envelopes_height = [x[1] for x in envelopes]
        return self.LIS(envelopes_height)

    def LIS(self, nums):
        """
        Longest increasing sequences
        """

        def binary_search(l, r, nums, target):
            while r - l > 1:
                m = l + (r - l) // 2
                if nums[m] >= target:
                    r = m
                else:
                    l = m
            return r

        if not nums:
            return 0

        size = len(nums)
        tail_list = [0] * (size + 1)
        LIS = 0
        tail_list[0] = nums[0]
        for i in range(size):
            if nums[i] < tail_list[0]:
                tail_list[0] = nums[i]
            elif nums[i] > tail_list[LIS - 1]:
                tail_list[LIS] = nums[i]
                LIS += 1
            else:
                tail_list[binary_search(-1, LIS - 1, tail_list,
                                        nums[i])] = nums[i]
        return LIS

# Another solution:
class Solution:
    def maxEnvelopes(self, envelopes):
        tails = []
        for w, h in sorted(envelopes, key = lambda x: (x[0], -x[1])):
            i = bisect.bisect_left(tails, h)
            if i == len(tails): tails.append(h)
            else: tails[i] = h
        return len(tails)

# 375 Guess Number Higher or Lower II
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

# 376 Wiggle Subsequence
class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 2:
            return len(nums)
        res = [nums[0]]
        for i in range(1, len(nums)):
            sign = (-1)**(len(res)-1)
            if nums[i]*sign > sign*res[-1]:
                res.append(nums[i])
            elif nums[i]*sign < sign*res[-1]:
                res[-1] = nums[i]
        max_length = len(res)
        res = [nums[0]]
        for i in range(1, len(nums)):
            sign = (-1)**(len(res))
            if nums[i]*sign > sign*res[-1]:
                res.append(nums[i])
            elif nums[i]*sign < sign*res[-1]:
                res[-1] = nums[i]
        max_length = max(max_length, len(res))
        return max_length

class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 2:
            return len(nums)
        up = [None] * len(nums)
        down = [None] * len(nums)
        up[0] = down[0] = 1
        for i in range(1, len(nums)):
            if nums[i]>nums[i-1]:
                up[i] = down[i-1] + 1
                down[i] = down[i-1]
            elif nums[i] < nums[i-1]:
                up[i] = up[i-1]
                down[i] = up[i-1]+1
            else:
                up[i] = up[i-1]
                down[i] = down[i-1]
        return max(up[-1], down[-1])

class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 2:
            return len(nums)
        # up = [None] * len(nums)
        #down = [None] * len(nums)
        up = down = 1
        for i in range(1, len(nums)):
            if nums[i]>nums[i-1]:
                up = down + 1
            elif nums[i] < nums[i-1]:
                down = up +1

        return max(up, down)

class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 2:
            return len(nums)
        # up = [None] * len(nums)
        #down = [None] * len(nums)
        pre_diff = nums[1]-nums[0]
        count = 2 if pre_diff !=0 else 1
        for i in range(2, len(nums)):
            diff = nums[i] - nums[i-1]
            if diff>0 and pre_diff <=0 or diff<0 and pre_diff >=0:
                count += 1
                pre_diff = diff
        return count

# 377 Combination Sum IV
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
# 392 Is Subsequence:
class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        def helper(ch, t):
            for i, t_ch in enumerate(t):
                # print(t_ch)
                if t_ch==ch:
                    return(i, t[i+1:])
            return (False, t)
        for ch in s:
            # print(ch)
            bool_, t = helper(ch, t)
            # print(bool_, t)
            if bool_ is False:
                return False
        return True

# 403 Frog Jump
class Solution(object):
    def canCross(self, stones):
        """
        :type stones: List[int]
        :rtype: bool
        """
        def helper(current, prev):
            if (current, prev) in memo:
                return memo[(current, prev)]
            if current == stones[-1]:
                memo[(current, prev)] = True
                return True
            if current > stones[-1]:
                memo[(current, prev)] = False
                return False
            if current not in stones:
                memo[(current, prev)] = False
                return False
            last_step = current - prev
            for step in [last_step-1, last_step, last_step+1]:
                if step >0 and helper(current+step, current):
                    memo[(current, prev)] = True
                    return True
            memo[(current, prev)] = False
            return False
        if not stones:
            return False
        if len(stones)==1:
            return True
        if stones[1]-stones[0] != 1:
            return False
        memo = {}
        # helper(stones[1], stones[0])
        # print(memo)
        return helper(stones[1], stones[0])

# 410 Split Array Larget Sum
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

# 413 Arithmetic slices:
"""
A sequence of number is called arithmetic if it consists of at least
three elements and if the difference between any two consecutive
elements is the same.

For example, these are arithmetic sequence:

1, 3, 5, 7, 9
7, 7, 7, 7
3, -1, -5, -9
The following sequence is not arithmetic.

1, 1, 2, 5, 7

A zero-indexed array A consisting of N numbers is given. A slice of
that array is any pair of integers (P, Q) such that 0 <= P < Q < N.

A slice (P, Q) of array A is called arithmetic if the sequence:
A[P], A[p + 1], ..., A[Q - 1], A[Q] is arithmetic. In particular,
this means that P + 1 < Q.

The function should return the number of arithmetic slices in the array A.


Example:

A = [1, 2, 3, 4]

return: 3, for 3 arithmetic slices in A: [1, 2, 3], [2, 3, 4] and
[1, 2, 3, 4] itself.
"""
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        def is_arithmetic(nums):
            """
            checking arithmetic for an array of length 3
            """
            return nums[2]-nums[1] == nums[1]-nums[0]
        def count_arithmetic_from_arithmetic_series(N):
            """
            count all of posible sub arithmetic that has length N
            So if first index start fom 0, we will have total N-3
            airthmetic (associcated with second index go from 2 to N-1)
            if first index start from 1, we will have N-2
            ....
            so total = (N-2) + (N-4) + ... + 1
            """
            return (N-2)*(N-3)//2

        if len(A) < 3:
            return 0
        p, q = 0, 2
        count = 0
        N = len(A)
        while p<N and q <N:
            if q-p>=2 and is_arithmetic(A[p:q+1]):
                count += 1
                while q+1<N and A[q+1]-A[q] == A[q]-A[q-1]:
                    q += 1
                    count += 1
                    # print(p, q, count)
                count += count_arithmetic_from_arithmetic_series(q+1-p)
                p = q-2
            if q == N-1 and p < N-3:
                p+=1

            else:
                p += 1
                q += 1
        return count

# Recursion:
"""
Observation: if A[i] - A[i-1] == A[i-1] - A[i-2] then we have additional
1+x arithmetric where x = slice(A, i-1) because (1, i) ...
(i-2, i) can be map to (0, i-1), (1, i-1), ..., (i-3, i-1) which count
equal to slice (A, i-1) = x. 1 more arithmetic is (0, i)
"""
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        self.count = 0
        def slices(A, i):
            """
            count number of arithmetic of matrix from A[0:i+1]

            """
            if i <2:
                return 0
            ap = 0
            if (A[i] - A[i-1] == A[i-1] - A[i-2]):
                ap = 1 + slices(A, i-1)
                self.count += ap
            else:
                slices(A, i-1)
            return ap


        slices(A, len(A)-1)
        return self.count

# Dynamic programing:
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        dp = [0] * len(A)
        count = 0
        for i in range(2, len(A)):
            if A[i]-A[i-1] == A[i-1]-A[i-2]:
                dp[i] = 1+ dp[i-1]
                count += dp[i]
        return count
#
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        dp = 0
        count = 0
        for i in range(2, len(A)):
            if A[i]-A[i-1] == A[i-1]-A[i-2]:
                dp = 1+dp
                count += dp
            else:
                dp=0
        return count

# 416 Partition equal subset sum
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

# backtracking list: 10, 17, 22, 37, 39, 40, 44, 46, 47, 51, 52, 60,
# 77, 78, 79, 89, 90, 93, 126, 131, 140, 211, 212, 216, 306, 357, 401,
# 526, 691, 784, 842, 980, 996

# 464 Can I win:
"""
The variable "cur" represents which numbers have been chosen
and which are not. For example, if maxChoosableInteger = 8, we have all
the numbers we can choose from: 1, 2, 3, 4, 5, 6, 7, 8.

The "cur" variable lets us know the state of our number stock.
Originally, cur = 0, or cur = 0000 0000 (in 8-bit system). This means,
all number from 1 to 8 have not been chosen.
At some point, let's say cur = 0000 0001, that means number 1 have been
picked by some player. Or when cur = 0000 0101, this means number 1 and
3 have been picked.

And so if (cur >> i) & 1 == 0 is checking if number (i+1) has been
picked yet or not, so we don't pick what has been picked (since the
problem said that we pick without replacement, which means you cannot
reuse a number). You shouldn't do what I did, just use hash set.
Looking back at this, I don't understand why I did this and not use a
hash set. @@
"""
class Solution(object):
    def canWin(self, maxChoosableInteger, desiredTotal, cur, d):
        if cur in d: return d[cur]
        if desiredTotal <= 0:
            d[cur] = False
            return d[cur]
        for i in range(maxChoosableInteger):
            if (cur >> i) & 1 == 0:
                if not self.canWin(maxChoosableInteger, desiredTotal - (i+1), cur + (1 << i), d):
                    d[cur] = True
                    return d[cur]
        d[cur] = False
        return d[cur]

    def canIWin(self, maxChoosableInteger, desiredTotal):
        if desiredTotal <= 0: return True
        if (maxChoosableInteger+1)*maxChoosableInteger/2 < desiredTotal: return False
        return self.canWin(maxChoosableInteger, desiredTotal, 0, {})

class Solution(object):
    def canIWin(self, max, desiredTotal):
        options = tuple(range(1,max+1))
        dp = {}
        def dfs(options, remain):
            if options[-1] >= remain or options in dp:
                return dp.setdefault(options,True)
            for i,j in enumerate(options):
                if not dfs(options[:i] + options[i+1:], remain-j):
                    return dp.setdefault(options,True)
            return dp.setdefault(options,False)
        return sum(options) >= desiredTotal and dfs(options, desiredTotal)

# 466 Count the rêptitions
"""
Define S = [s,n] as the string S which consists of n connected strings s.
For example, ["abc", 3] ="abcabcabc".

On the other hand, we define that string s1 can be obtained from string
s2 if we can remove some characters from s2 such that it becomes s1.
For example, “abc” can be obtained from “abdbec” based on our definition,
but it can not be obtained from “acbbe”.

You are given two non-empty strings s1 and s2 (each at most 100 characters
long) and two integers 0 ≤ n1 ≤ 106 and 1 ≤ n2 ≤ 106. Now consider the s
trings S1 and S2, where S1=[s1,n1] and S2=[s2,n2]. Find the maximum
integer M such that [S2,M] can be obtained from S1.

Example:

Input:
s1="acb", n1=4
s2="ab", n2=2

Return:
2
"""
class Solution(object):
    def getMaxRepetitions(self, s1, n1, s2, n2):
        """
        :type s1: str
        :type n1: int
        :type s2: str
        :type n2: int
        :rtype: int
        """
        count = 0
        index = 0
        indexr = [0] *(len(s2)+1)
        countr = [0] *(len(s2)+1)

        for i in range(0, n1):
            for j in range(0, len(s1)):

                if s1[j] == s2[index]:
                    index += 1
                if index == len(s2):
                    index = 0
                    count += 1
            countr[i] = count
            indexr[i] = index
            for k in range(0, i):
                if index == indexr[k]:
                    prev_count = countr[k]
                    pattern_count = (countr[i]-countr[k])*((n1-1-k)//(i-k))
                    remain_count = countr[k + (n1-1-k)% (i-k)] - countr[k]
                    # print(k, i, countr, indexr, count, index)
                    return (prev_count + pattern_count + remain_count) // n2
        return countr[n1-1]//n2

# 467 Unique Substrign in Wraparound String
"""
Consider the string s to be the infinite wraparound string of
"abcdefghijklmnopqrstuvwxyz", so s will look like this:
"...zabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcd....".

Now we have another string p. Your job is to find out how many unique
non-empty substrings of p are present in s. In particular, your input
is the string p and you need to output the number of different non-empty
substrings of p in the string s.

Note: p consists of only lowercase English letters and the size of p
might be over 10000.

Example 1:
Input: "a"
Output: 1

Explanation: Only the substring "a" of string "a" is in the string s.
Example 2:
Input: "cac"
Output: 2
Explanation: There are two substrings "a", "c" of string "cac" in the
string s.
Example 3:
Input: "zab"
Output: 6
Explanation: There are six substrings "z", "a", "b", "za", "ab", "zab"
 of string "zab" in the string s.
"""
class Solution(object):
    def findSubstringInWraproundString(self, p):
        """
        :type p: str
        :rtype: int
        """
        sequence_len = 0
        max_len_end_at = {}
        count = 0
        for i, c in enumerate(p):
            if i>0 and ord(p[i])-ord(p[i-1]) in [1, -25]:
                sequence_len += 1
            else:
                sequence_len = 1
            if c not in max_len_end_at:
                count += sequence_len
                max_len_end_at[c] = sequence_len
            elif sequence_len > max_len_end_at[c]:
                count += sequence_len - max_len_end_at[c]
                max_len_end_at[c] = sequence_len
            else:
                pass

        return count



"""
Given an integer matrix, find the length of the longest increasing path.

From each cell, you can either move to four directions: left, right,
up or down. You may NOT move diagonally or move outside of the boundary
(i.e. wrap-around is not allowed).

Example 1:

Input: nums =
[
  [9,9,4],
  [6,6,8],
  [2,1,1]
]
Output: 4
Explanation: The longest increasing path is [1, 2, 6, 9].
Example 2:

Input: nums =
[
  [3,4,5],
  [3,2,6],
  [2,2,1]
]
Output: 4
Explanation: The longest increasing path is [3, 4, 5, 6]. Moving
diagonally is not allowed.
"""

class Solution(object):
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        if not matrix:
            return 0
        graph, incoming_deg = collections.defaultdict(set), collections.defaultdict(int)

        for i, row in enumerate(matrix):
            for j, cell in enumerate(row):
                if i>0 and matrix[i-1][j] > cell:
                    graph[(i, j)] |= {(i-1, j)}
                    incoming_deg[(i-1, j)] += 1
                if i+1<len(matrix) and matrix[i+1][j]>cell:
                    graph[(i, j)] |= {(i+1, j)}
                    incoming_deg[(i+1, j)] +=1

                if j>0 and matrix[i][j-1] > cell:
                    graph[(i, j)] |= {(i, j-1)}
                    incoming_deg[(i, j-1)] += 1
                if j+1<len(row) and matrix[i][j+1]>cell:
                    graph[(i, j)] |= {(i, j+1)}
                    incoming_deg[(i, j+1)] +=1

        max_length = 0
        S = collections.deque()
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if not incoming_deg[(i, j)]:
                    S.append(((i,j), 1))

        while S:
            cur_node, cur_length = S.popleft()
            max_length = max(max_length, cur_length)
            for adj_node in graph[cur_node]:
                incoming_deg[adj_node] -= 1
                if incoming_deg[adj_node] == 0:
                    S.append((adj_node, cur_length+1))
        return max_length

# DP solutions
class Solution:
    def longestIncreasingPath(self, matrix):
        if not matrix:
            return 0
        res=0
        m=len(matrix)
        n=len(matrix[0])
        dp=[[0 for i in range(n)]for j in range(m)]
        for i in range(m):
            for j in range(n):
                tem=self.dfs(matrix,dp,i,j,n,m)
                res=max(res,tem)
        return res
    def dfs(self,matrix,dp,i,j,n,m):
        if not dp[i][j]:
            dp[i][j]=1+max(self.dfs(matrix,dp,i-1,j,n,m)if i-1>=0 and matrix[i-1][j]<matrix[i][j] else 0,
                         self.dfs(matrix,dp,i+1,j,n,m)if i+1<m and matrix[i+1][j]<matrix[i][j] else 0,
                         self.dfs(matrix,dp,i,j+1,n,m)if j+1<n and matrix[i][j+1]<matrix[i][j] else 0,
                         self.dfs(matrix,dp,i,j-1,n,m)if j-1>=0 and matrix[i][j-1]<matrix[i][j] else 0)
        return dp[i][j]

class Solution(object):
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        if not matrix or not matrix[0]:
            return 0

        row,col = len(matrix),len(matrix[0])
        num_coordinate = set([(matrix[i][j],i,j) for i in xrange(row) for j in xrange(col)])
        nums_tuple = sorted(list(num_coordinate),key = lambda x:x[0])

        dp = [[1 for _ in xrange(col)] for _ in xrange(row)]
        steps = [(0,1),(0,-1),(1,0),(-1,0)]
        res = 1
        for i in xrange(len(nums_tuple)):
            cur_num,x,y = nums_tuple[i]
            for dx,dy in steps:
                new_x,new_y = x + dx,y + dy
                if new_x < 0 or new_x >= row or new_y < 0 or new_y >= col:
                    continue
                if matrix[new_x][new_y] >= cur_num:
                    continue
                dp[x][y] = max(dp[x][y],dp[new_x][new_y] + 1)
                res = max(res,dp[x][y])
        return res


# Depth First Search + Memoization
class Solution(object):
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        if not matrix or not matrix[0]:
            return 0

        res = float('-inf')
        row,col = len(matrix),len(matrix[0])
        memo = {}
        for i in xrange(row):
            for j in xrange(col):
                temp_len = self.check(matrix,i,j,memo)  # if already (i,j) in memo, directly return
                res = max(res,temp_len)
        return res

    def check(self,matrix,i,j,memo):
        if (i,j) in memo:
            return memo[(i,j)]
        max_len = 0
        row,col = len(matrix),len(matrix[0])
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            x,y = i + dx,j + dy
            if x < 0  or x >= row or y < 0 or y >= col or matrix[i][j] <= matrix[x][y]:
                continue
            tmp_len = self.check(matrix,x,y,memo)
            max_len = max(tmp_len,max_len)
        memo[(i,j)] = max_len + 1
        return max_len + 1

# 330 Patching array:
"""
Given a sorted positive integer array nums and an integer n, add/patch
elements to the array such that any number in range [1, n] inclusive
can be formed by the sum of some elements in the array. Return the
minimum number of patches required.

Example 1:

Input: nums = [1,3], n = 6
Output: 1
Explanation:
Combinations of nums are [1], [3], [1,3], which form possible sums of: 1, 3, 4.
Now if we add/patch 2 to nums, the combinations are: [1], [2], [3], [1,3], [2,3], [1,2,3].
Possible sums are 1, 2, 3, 4, 5, 6, which now covers the range [1, 6].
So we only need 1 patch.
Example 2:

Input: nums = [1,5,10], n = 20
Output: 2
Explanation: The two patches can be [2, 4].
Example 3:

Input: nums = [1,2,2], n = 5
Output: 0
"""
class Solution(object):
    def minPatches(self, nums, n):
        """
        :type nums: List[int]
        :type n: int
        :rtype: int
        """
        count = 0
        patch = 0
        i = 0
        while patch < n:
            if i < len(nums) and patch + 1 >= nums[i]:
                patch += nums[i]
                i += 1
            else:
                count += 1
                patch += patch + 1
        return count

# 334 Increasing triple sebsequence
"""
Given an unsorted array return whether an increasing subsequence of
length 3 exists or not in the array.

Formally the function should:

Return true if there exists i, j, k
such that arr[i] < arr[j] < arr[k] given 0 ≤ i < j < k ≤ n-1 else return false.
Note: Your algorithm should run in O(n) time complexity and O(1) space complexity.

Example 1:

Input: [1,2,3,4,5]
Output: true
Example 2:

Input: [5,4,3,2,1]
Output: false
"""

class Solution(object):
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(nums) < 3:
            return False
        num_2, num_1 = None, nums[0]
        for i, num in enumerate(nums):
            if num_2 is not None and num>num_2:
                return True
            if num > num_1:
                num_2 = num
            if num < num_1:
                num_1 = num
            print(num, num_1, num_2)
        return False
