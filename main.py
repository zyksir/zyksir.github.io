
MAX_N = int(1e9)
def calculate_distance(a, x, n):
    dist = 0
    for i in range(n):
        dist += abs(a[i] - x)
    return dist

def solution1(a, n, d):
    count = 0
    for x in range(a[0] - d, a[-1] + d + 1):
        if calculate_distance(a, x, n) * 2 <= d:
            count += 1
    return count

def binary_check_for_mono_increase(left, right, a, n, d):
    ans = left
    while left <= right:
        mid = (left + right) // 2
        if calculate_distance(a, mid, n) * 2 <= d:
            ans = max(ans, mid)
            left = mid + 1
        else:
            right = mid - 1
    return ans

# 对于单调递减函数, 要找到最小的left
def binary_check_for_mono_decrease(left, right, a, n, d):
    ans = right
    while left <= right:
        mid = (left + right) // 2
        if calculate_distance(a, mid, n) * 2 <= d:
            ans = min(ans, mid)
            right = mid - 1
        else:
            left = mid + 1
    return ans

def solution2(a, n, d):
    lowest_point = a[n//2]
    if calculate_distance(a, lowest_point, n) * 2 > d:
        return 0
    left_bound = binary_check_for_mono_decrease(a[0] - d, lowest_point, a, n, d)
    right_bound = binary_check_for_mono_increase(lowest_point, a[-1] + d, a, n, d)
    return right_bound - left_bound + 1

def solution3(a, n, d):
    F = [calculate_distance(a, a[0], n)]
    for i in range(n-1):
        F.append( (2*(i+1) - n) * (a[i+1] - a[i]) + F[i] )
    l, r = 0, n-1
    while l <= r:
        if 2*F[l] <= d and 2*F[r] <= d:
            break
        l += 1
        r -= 1
    if l > r:
        return 0
    return a[r] - a[l] + 1 + (d//2-F[r])//(r - l + 1) + (d//2-F[l])//(r - l + 1)

a, d = [-2, 0, 1], 8
n = len(a)
print(solution1(a, n, d))
print(solution2(a, n, d))
print(solution3(a, n, d))