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
        for i in xrange(index, len(nums)):
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
            return self.isNumber(s[0:ind_e]) and self.isNumber(s[ind_e+1:]) and ('.' not in s[ind_e+1:]) and ind_e>0 and ind_e<len(s)-1
        
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

#121 Best time to buy and sell stock II
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