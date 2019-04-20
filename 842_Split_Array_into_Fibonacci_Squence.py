"""
Given a string S of digits, such as S = "123456579", we can split it
into a Fibonacci-like sequence [123, 456, 579].

Formally, a Fibonacci-like sequence is a list F of non-negative integers
such that:

0 <= F[i] <= 2^31 - 1, (that is, each integer fits a 32-bit signed
integer type);
F.length >= 3;
and F[i] + F[i+1] = F[i+2] for all 0 <= i < F.length - 2.
Also, note that when splitting the string into pieces, each piece must
not have extra leading zeroes, except if the piece is the number 0 itself.

Return any Fibonacci-like sequence split from S, or return [] if it
cannot be done.

Example 1:

Input: "123456579"
Output: [123,456,579]
Example 2:

Input: "11235813"
Output: [1,1,2,3,5,8,13]
Example 3:

Input: "112358130"
Output: []
Explanation: The task is impossible.
Example 4:

Input: "0123"
Output: []
Explanation: Leading zeroes are not allowed, so "01", "2", "3" is not valid.
Example 5:

Input: "1101111"
Output: [110, 1, 111]
Explanation: The output [11, 0, 11, 11] would also be accepted.
"""
class Solution(object):
    def splitIntoFibonacci(self, S):
        """
        :type S: str
        :rtype: List[int]
        """
        def backtracking(pos, current):
            if pos==len(S):
                if len(current) >=3:
                    self.ans = current
                    return True
                else:
                    return False
            first = current[-2]
            second = current[-1]
            if S[pos] == 'O' and first+second>0:
                # if the next character is 0 meaning the next number
                # must be 0 so, if first + second >0 -> False
                return False

            if first+second>2**31-1:
                return False

            new = str(first+second)
            len_new = len(new)
            current.append(first+second)
            if S[pos: pos+len_new] == new:
                return backtracking(pos+len_new, current)

        for i in range(1, min(len(S)//3+2, 11)):
            for j in range(1, min((len(S)-i)//2+1, 11)):
                if S[0] =='0' and i>1:
                    continue
                if S[i] =='0' and j>1:
                    continue
                first = int(S[:i])
                second = int(S[i:i+j])
                # print(first, second, i, j)
                if first > 2**31-1 or second>2**31-1:
                    continue
                current = [first, second]
                if backtracking(i+j, current):
                    return self.ans

# Solution 2:
class Solution(object):
    def splitIntoFibonacci(self, S):
        """
        :type S: str
        :rtype: List[int]
        """
        length = len(S)
        if length < 3: return []
        for i in range(1, min(length//2+1, 11)):
            for j in range(1, min(length//2+1, 11)):
                temp = []
                if S[0]=='0' and i!=1:
                    continue
                if S[i]=='0' and j!=1:
                    continue
                temp.append(int(S[0:i]))
                temp.append(int(S[i:i+j]))
                #print(temp)
                #print(i,j)
                k = i+j
                while(k < length):
                    flag = False
                    t_sum = temp[-1]+temp[-2]
                    if t_sum > 2147483647:
                        break
                    sp = str(t_sum)
                    sp_len = len(sp)
                    if k+sp_len > length or S[k:k+sp_len]!=sp:
                        break
                    else:
                        k = k+sp_len
                        temp.append(t_sum)
                    if k==length:
                        return temp
        return []
