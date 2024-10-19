from collections import Counter

Counter("string")


list1 = [1, 3, 5]
list2 = [2, 4, 6]

sorted(list1 + list2)

help(sorted)


def merge_sorted_lists(list1, list2):
    merged = []
    i = j = 0

    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1

    merged.extend(list1[i:])
    merged.extend(list2[j:])

    return merged


merge_sorted_lists(list1, list2)

arr = [1, 2, 3, 4, 5]


# It returns location of x in given array arr
def binarySearch(arr, low, high, x):

    while low <= high:

        mid = low + (high - low) // 2

        # Check if x is present at mid
        if arr[mid] == x:
            return mid

        # If x is greater, ignore left half
        if arr[mid] < x:
            low = mid + 1

        # If x is smaller, ignore right half
        else:
            high = mid - 1

    # If we reach here, then the element
    # was not present
    return -1


binarySearch(arr, 0, len(arr) - 1, 5)


def find_unique_char(char: str):
    char_dict = {}
    for c in set(char):
        char_dict[c] = list(char).count(c)

    for key, value in char_dict.items():
        if value == 1:
            print(
                f"the first non-repeating caharacter is {key} at index {list(char).index(key)}",
            )
            break
    else:
        print("-1 (All characters repeated)")


find_unique_char("guess")


def is_balanced(s):

    # Stack to keep track of opening brackets
    stack = []

    # Dictionary to map closing brackets to their corresponding opening brackets
    matching_bracket = {")": "(", "}": "{", "]": "["}

    # Iterate over each character in the string
    for char in s:
        if char in "({[":  # If it's an opening bracket, push to the stack
            stack.append(char)
        elif char in ")}]":  # If it's a closing bracket
            if not stack or stack[-1] != matching_bracket[char]:
                return False  # Unbalanced if there's no matching opening bracket
            stack.pop()  # Pop the matching opening bracket

    # If the stack is empty, all brackets were matched and balanced
    return not stack


# Example usage
input_string = "{[()()]"  # You can change the input to test other cases
print(is_balanced(input_string))  # Output: True for this balanced example


def reserve_string(char: str):
    reverse_char = ""
    for i in range(1, len(char) + 1):
        # print(char[-i])
        reverse_char += char[-i]
    return reverse_char


reserve_string("bbq")

num = 9


def is_prime(num: int):
    for i in range(2, num):
        if num % i == 0:
            print(False)
            break
    else:
        print(True)


is_prime(90)


def is_palindrome(char: str):
    reserse_char = reserve_string(char)
    if reserse_char == char:
        print(True)
    else:
        print(False)


is_palindrome("hello")

set("hello")

"hello".count("h")


def find_pairs(list_name: list, value: int):
    results = []
    for i in range(len(list_name)):
        for j in range(i + 1, len(list_name)):
            if list_name[i] + list_name[j] == value:
                results.append((list_name[i], list_name[j]))
    return results


find_pairs([2, 4, 3, 5, 7, 0], 7)

n = 5
k = 1
for i in range(1, n + 1):
    k *= i
k
