import math
import numpy as np

## 1a
a=123
b=321
print (a*b)

## 1b
print ("\nvec1:")
vec1 = [3, 8, 9, 10, 12]
print (vec1)

print ("\nvec2:")
vec2 = [8, 7, 7, 5, 6]
print (vec2)

print ("\nsuma wektorow:")
print ([x+y for x,y in zip (vec1, vec2)])

print ("\niloczyn wektorow (po wspolrzednych):")
vec3 = [x*y for x,y in zip (vec1, vec2)]
print (vec3)

## 1c
print ("\niloczyn skalarny:")
sum = 0
for x in vec3:
    sum = sum + x
print (sum)

print ("\ndlugosc vec1:")
vec1_len = 0
for x in vec1:
    vec1_len = x * x + vec1_len
vec1_len = math.sqrt (vec1_len)
print (vec1_len)


print ("\ndlugosc vec2:")
vec2_len = 0
for x in vec2:
    vec2_len = x * x + vec2_len
vec2_len = math.sqrt (vec2_len)
print (vec2_len)

## 1d

mat1 = np.matrix ('1 2 3; 2 2 2; 4 1 7')
print ("\nmat1:")
print (mat1)
mat2 = np.matrix ('4 5 6; 4 5 6; 4 5 6')
print ("\nmat2:")
print (mat2)

mat3 = np.multiply (mat1, mat2)
print ("\nmnozenie po wspolrzednych:")
print (mat3)

mat3 = np.dot(mat1, mat2)
print ("\nmnozenie wektorowe")
print (mat3)

## 1e
vec4 = np.random.randint(low=1, high=100, size=50)
print ("\n50 losowych liczb (vec4):")
print (vec4)

## 1f
min_vec4 = np.min (vec4)
max_vec4 = np.max (vec4)

print ("\nmin(vec4):")
print (min_vec4)
print ("\nmax(vec4):")
print (max_vec4)
print ("\nmean(vec4):")
print (np.mean(vec4))
print ("\nstd(vec4):")
print (np.std(vec4))

for i, x in np.ndenumerate(vec4):
    if vec4[i] == max_vec4:
        print ("\nindex for max in vec4: " + str(i))
        ind_max_vec4 = i

    if vec4[i] == min_vec4:
        print ("\nindex for min in vec4: " + str(i))
        ind_min_vec4 = i
## 1g
vec5 = np.zeros (shape=50)
for i, x in np.ndenumerate(vec4):
    vec5[i] = (vec4[i] - min_vec4) / (max_vec4 - min_vec4)

print ("\n vec5:")
print (vec5)

print ("\n under the max_index in new vector: " + str(vec5[ind_max_vec4[0]]))
