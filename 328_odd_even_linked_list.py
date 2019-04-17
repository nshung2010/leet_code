"""
Given a singly linked list, group all odd nodes together followed by
the even nodes. Please note here we are talking about the node number
and not the value in the nodes.

You should try to do it in place. The program should run in O(1)
space complexity and O(nodes) time complexity.

Example 1:

Input: 1->2->3->4->5->NULL
Output: 1->3->5->2->4->NULL
Example 2:

Input: 2->1->3->5->6->4->7->NULL
Output: 2->3->6->7->1->5->4->NULL
"""
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        tail = head
        length = 1
        while tail.next:
            tail = tail.next
            length += 1
        if length<=2:
            return head
        odd = head
        even_part = None
        for _ in range(length//2):
            even = odd.next
            temp = even.next
            even.next = None
            tail.next = even
            tail=even
            odd.next=temp
            odd = temp
        return head

# Solution 2
class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # corner cases of [] and [1]
        if not head or not head.next: return head
        odd_tail, even_head, even_tail = head, head.next, head.next
        # maintain 3 pointers:
        # odd_tail: tail node in the processed odd list
        # even_head: first even node that'll be connected to odd_tail at termination
        # even_tail: tail of even node list, so that even_tail.next is odd node or none
        # when even_tail or even_tail.next becomes none, terminate and connect
        while even_tail and even_tail.next:
            odd_temp  = even_tail.next
            odd_tail.next, odd_tail = odd_temp, odd_temp
            # disengage odd and even list, link up only when terminating
            even_tail.next = odd_temp.next
            even_tail = even_tail.next
        odd_tail.next = even_head
        return head
