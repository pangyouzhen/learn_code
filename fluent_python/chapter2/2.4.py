import bisect


#  bisect 的应用场景

# 根据学生成绩进行分级 学生成绩(0-100)，分级成(ABCDEF)：

def grade(score):
    if score < 60:
        return "F"
    elif score < 70:
        return 'E'
    elif score < 80:
        return 'D'
    elif score < 90:
        return 'C'
    elif score < 100:
        return 'B'
    else:
        return "A"


scores = [59, 60, 72, 82, 85, 90, 98, 100]
result = [grade(x) for x in scores]
print(result)


def bisect_grade(score, points=None, grade="FEDCBA"):
    if points is None:
        points = [60, 70, 80, 90, 100]
    item = bisect.bisect(points, score)
    return grade[item]


bisect_result = [bisect_grade(x) for x in scores]
print(bisect_result)
