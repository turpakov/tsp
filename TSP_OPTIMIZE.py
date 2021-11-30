import csv
import sys
import numpy as np

def findSmallestElementInRow(matrix, row):
    minElement = sys.maxsize
    for element in matrix[row]:
        if minElement > element:
            minElement = element
    return minElement


def findSmallestElementInCol(matrix, col):
    minElement = sys.maxsize
    for element in matrix[:, col]:
        if minElement > element:
            minElement = element
    return minElement


def rowReduction(matrix):
    rowCoefficients = []
    for i in range(len(matrix)):
        minElement = findSmallestElementInRow(matrix, i)
        rowCoefficients.append(minElement)
        for j in range(len(matrix[i])):
            matrix[i][j] -= minElement
    return matrix, rowCoefficients


def colReduction(matrix):
    colCoefficients = []
    for i in range(len(matrix)):
        minElement = findSmallestElementInCol(matrix, i)
        colCoefficients.append(minElement)
        for j in range(len(matrix[:, i])):
            matrix[j][i] -= minElement
    return matrix, colCoefficients


def evaluationCalculation(matrix, cur_way):
    maxAssessment = -1
    maxI = -1
    maxJ = -1
    row = [cur_way[i] for i in range(len(cur_way)) if i % 2 == 0]
    col = [cur_way[i] for i in range(len(cur_way)) if i % 2 == 1]
    for i in range(len(matrix)):
        if i not in row:
            for j in range(len(matrix[i])):
                if j not in col:
                    if matrix[i][j] == 0:
                        minRow = sys.maxsize
                        minCol = sys.maxsize
                        for k in range(len(matrix[i])):
                            if matrix[i][k] < minRow and k != j:
                                minRow = matrix[i][k]
                        for k in range(len(matrix[:, j])):
                            if matrix[k][j] < minCol and k != i:
                                minCol = matrix[k][j]
                        sumOfElements = minCol + minRow
                        if sumOfElements > maxAssessment:
                            maxAssessment = sumOfElements
                            maxI = i
                            maxJ = j
    return maxI, maxJ, maxAssessment


def cutMatrix(matrix, indexI, indexJ):
    """
    Inf по индексам строки i и столбца j
    """
    matrix[indexI, :] = np.array([float('inf') for elem in matrix[indexI, :]])
    matrix[:, indexJ] = np.array([float('inf') for elem in matrix[:, indexJ]])
    matrix[indexI][indexJ] = 0
    return matrix

def bestRecordIdx(tree):
    z = sys.maxsize
    idx = 0
    for i in range(len(tree)):
        if tree[i][0] < z:
            z = tree[i][0]
            idx = i
    min_lst = []
    for j in range(len(tree)):
        if tree[j][0] == z:
            min_lst.append(j)
    l = -1
    for i in min_lst:
        if len(tree[i][2]) > l:
            l = len(tree[i][2])
            idx = i
    return idx

def full_path(ids):
    result = []
    result.append(ids[0][0])
    result.append(ids[0][1])
    ids.remove(ids[0])
    i = 0
    while ids:
        if ids[i][0] == result[-1]:
            result.append(ids[i][1])
            ids.remove(ids[i])
            i = 0
            continue
        i += 1
    return result

def potencial_circle(cur_way):
    pairs = []
    for i in range(0, len(cur_way) - 1, 2):
        pairs.append([cur_way[i], cur_way[i + 1]])
    all_circles = []
    for edge1 in pairs:
        i = 0
        buff = edge1.copy()
        end = edge1[1]
        f = True
        while i != len(pairs):
            if end == pairs[i][0]:
                buff.append(pairs[i][0])
                buff.append(pairs[i][1])
                if buff not in all_circles:
                    all_circles.append(buff)
                    buff = edge1.copy()
                    end = buff[1]
                    i = 0
                    continue
                f = False
                end = pairs[i][1]
                i = 0
            else:
                i += 1
        if len(buff) > 2 and f:
            all_circles.append(buff)
    result = []
    for circle in all_circles:
        result.append([circle[-1], circle[0]])
    return result

def Is_the_end(matrix):
    ids = []
    row = []
    col = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] == 0:
                if i not in row and j not in col:
                    ids.append([i, j])
                    row.append(i)
                    col.append(j)
                else:
                    return False, None
            elif matrix[i][j] != float('inf'):
                return False, None
    return True, ids

def TSP_LITTLE(matrix):
    tree = []
    cur_matrix = matrix.copy()
    cur_matrix, row_coeff = rowReduction(cur_matrix)
    cur_matrix, col_coeff = colReduction(cur_matrix)
    idxI, idxJ, maxAssessment = evaluationCalculation(cur_matrix, [])
    left_matrix = cutMatrix(cur_matrix, idxI, idxJ)
    left_matrix[idxJ][idxI] = float("inf")
    Y_l = sum(row_coeff) + sum(col_coeff)

    right_matrix = matrix.copy()
    right_matrix[idxI][idxJ] = float("inf")
    right_matrix, row_coeff_r = rowReduction(right_matrix)
    right_matrix, col_coeff_r = colReduction(right_matrix)
    Y_r = Y_l + maxAssessment

    tree.append((Y_l, [idxJ, idxI], [idxI, idxJ], row_coeff, col_coeff))
    tree.append((Y_r, [idxI, idxJ], [], row_coeff_r, col_coeff_r))

    while True:
        best_id = bestRecordIdx(tree)
        cur_sol = tree[best_id]
        z = cur_sol[0]
        cur_matrix = matrix.copy()
        invalid_elem = cur_sol[1]
        invalid_elem_r = invalid_elem.copy()
        invalid_elem_l = invalid_elem.copy()
        cur_way = cur_sol[2].copy()
        row_coeff = cur_sol[3].copy()
        col_coeff = cur_sol[4].copy()
        cur_way_l = cur_way.copy()
        for i in range(len(row_coeff)):
            cur_matrix[i, :] = np.array([elem - row_coeff[i] for elem in cur_matrix[i, :]])
        for j in range(len(col_coeff)):
            cur_matrix[:, j] = np.array([elem - col_coeff[j] for elem in cur_matrix[:, j]])
        for i in range(0, len(cur_way) - 1, 2):
            cur_matrix = cutMatrix(cur_matrix, cur_way[i], cur_way[i+1])
        for i in range(0, len(invalid_elem) - 1, 2):
            cur_matrix[invalid_elem[i]][invalid_elem[i+1]] = float('inf')
        right_matrix = cur_matrix.copy()
        del tree[best_id]
        Y = z

        if Is_the_end(cur_matrix)[0]:
            idx = Is_the_end(cur_matrix)[1]
            result_way = full_path(idx)
            return z, result_way, len(result_way)-1

        idxI, idxJ, maxAssessment = evaluationCalculation(cur_matrix, cur_way)
        cur_way_l.append(idxI)
        cur_way_l.append(idxJ)
        left_matrix = cutMatrix(cur_matrix, idxI, idxJ)
        left_matrix[idxJ][idxI] = float("inf")
        potencial_circle_edge = potencial_circle(cur_way_l)
        if potencial_circle_edge:
            for circle in potencial_circle_edge:
                inv_idxI = circle[0]
                inv_idxJ = circle[1]
                left_matrix[inv_idxI][inv_idxJ] = float("inf")
                invalid_elem_l.append(inv_idxI)
                invalid_elem_l.append(inv_idxJ)

        left_matrix, row_coeff_l = rowReduction(left_matrix)
        left_matrix, col_coeff_l = colReduction(left_matrix)
        invalid_elem_l.append(idxJ)
        invalid_elem_l.append(idxI)
        row_coeff_l = [row_coeff[i] + row_coeff_l[i] for i in range(len(row_coeff))]
        col_coeff_l = [col_coeff[i] + col_coeff_l[i] for i in range(len(col_coeff))]
        Y_l = sum(row_coeff_l) + sum(col_coeff_l)

        invalid_elem_r.append(idxI)
        invalid_elem_r.append(idxJ)
        right_matrix[idxI][idxJ] = float("inf")
        right_matrix, row_coeff_r = rowReduction(right_matrix)
        right_matrix, col_coeff_r = colReduction(right_matrix)
        row_coeff_r = [row_coeff[i] + row_coeff_r[i] for i in range(len(row_coeff))]
        col_coeff_r = [col_coeff[i] + col_coeff_r[i] for i in range(len(col_coeff))]
        Y_r = Y + maxAssessment

        tree.append((Y_l, invalid_elem_l, cur_way_l, row_coeff_l, col_coeff_l))
        tree.append((Y_r, invalid_elem_r, cur_way, row_coeff_r, col_coeff_r))


# matrix = np.array([[float('inf'), 4, 5, 2, 9],
#                    [3, float('inf'), 3, 4, 8],
#                    [6, 4, float('inf'), 1, 7],
#                    [3, 2, 5, float('inf'), 8],
#                    [8, 9, 6, 8, float('inf')]])

# matrix = np.array([[float('inf'), 4, 6, 2, 9],
#                    [4, float('inf'), 3, 2, 9],
#                    [6, 3, float('inf'), 5, 9],
#                    [2, 2, 5, float('inf'), 8],
#                    [9, 9, 9, 8, float('inf')]])

# matrix = np.array([[float('inf'), 3, 93, 13, 33, 9, 57],
#                    [4, float('inf'), 77, 42, 21, 16, 34],
#                    [45, 17, float('inf'), 36, 16, 28, 25],
#                    [39, 90, 80, float('inf'), 56, 7, 91],
#                    [28, 46, 88, 33, float('inf'), 25, 57],
#                    [3, 88, 18, 46, 92, float('inf'), 7],
#                    [44, 26, 33, 27, 84, 39, float('inf')]])

# n = 100
# matrix = np.zeros([n, n])
# for i in np.arange(0, n, 1):
#     for j in np.arange(0, n, 1):
#         if i != j:
#             c = np.random.randint(0, 2)
#             matrix[i, j] = c
#         else:
#             matrix[i, j] = float('inf')



# cur_matrix = []
# with open(f"data/1.csv", "r") as rf:
#         file_reader = csv.reader(rf, delimiter=",")
#         for row in file_reader:
#             cur_matrix.append([float(i) for i in row])
# matrix = np.array(cur_matrix)

matrix = np.loadtxt("matrix_100_for_automat.txt", dtype="float")

answer = TSP_LITTLE(matrix)
print(f"Длина маршрута: {answer[0]}")
print(f"Маршрут: {answer[1]}")
print(f"Количество посещенных вершин: {answer[2]}")
