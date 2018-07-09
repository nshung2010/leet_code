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