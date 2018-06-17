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