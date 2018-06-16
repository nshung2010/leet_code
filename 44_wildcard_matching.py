class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
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