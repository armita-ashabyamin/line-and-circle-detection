import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt


#reading our input image:
input_image = cv.imread(r'F:\cv\Line-circle.png',0)

#converting our input image into binary
binr = cv.threshold(input_image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

#reading our circles kernel
input_kernel = cv.imread(r"F:\cv\kernel.jpg",0)
kernel = cv.threshold(input_kernel, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]



#detecting the circles :
erosion = cv.erode(binr, kernel, iterations=1)

dilation = cv.dilate(erosion, kernel, iterations=2)

opening = cv.morphologyEx(binr, cv.MORPH_OPEN, kernel, iterations=1)

#saving the circles pic
cv.imwrite('circles.jpg', opening)
circles = cv.imread(r"F:\cv\circles.jpg",0)

#circles have been detected


#finding our lines :

#defining our line kernels:
Line_kernel1 = cv.imread(r"F:\cv\Line_kernel1.jpg",0)

line_kernel1 = cv.threshold(Line_kernel1, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

Line_kernel2 = cv.imread(r"F:\cv\Line_kernel2.jpg",0)

line_kernel2 = cv.threshold(Line_kernel2, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

Line_kernel3 = cv.imread(r"F:\cv\Line_kernel3.jpg",0)

line_kernel3 = cv.threshold(Line_kernel3, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

Line_kernel4 = cv.imread(r"F:\cv\Line_kernel4.jpg",0)

line_kernel4 = cv.threshold(Line_kernel4, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

Line_kernel5 = cv.imread(r"F:\cv\Line_kernel5.jpg",0)

line_kernel5 = cv.threshold(Line_kernel5, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

Line_kernel6 = cv.imread(r"F:\cv\Line_kernel6.jpg",0)

line_kernel6 = cv.threshold(Line_kernel6, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

Line_kernel7 = cv.imread(r"F:\cv\Line_kernel7.jpg",0)

line_kernel7 = cv.threshold(Line_kernel7, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

Line_kernel8 = cv.imread(r"F:\cv\Line_kernel8.jpg",0)

line_kernel8 = cv.threshold(Line_kernel8, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

Line_kernel9 = cv.imread(r"F:\cv\Line_kernel9.jpg",0)

line_kernel9 = cv.threshold(Line_kernel9, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

Line_kernel10 = cv.imread(r"F:\cv\Line_kernel10.jpg",0)

line_kernel10 = cv.threshold(Line_kernel10, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

Line_kernel11 = cv.imread(r"F:\cv\Line_kernel11.jpg",0)

line_kernel11 = cv.threshold(Line_kernel11, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

Line_kernel12 = cv.imread(r"F:\cv\Line_kernel12.png",0)

line_kernel12 = cv.threshold(Line_kernel12, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]


#using our kernels to find the lines:

opening_kernel1 = cv.morphologyEx(binr, cv.MORPH_OPEN, line_kernel1, iterations=1)
closing_kernel1 = cv.morphologyEx(opening_kernel1, cv.MORPH_CLOSE, kernel, iterations=1)
opening_kernel2 = cv.morphologyEx(binr, cv.MORPH_OPEN, line_kernel2, iterations=1)
closing_kernel2 = cv.morphologyEx(opening_kernel2, cv.MORPH_CLOSE, kernel, iterations=1)
opening_kernel3 = cv.morphologyEx(binr, cv.MORPH_OPEN, line_kernel3, iterations=1)
closing_kernel3 = cv.morphologyEx(opening_kernel3, cv.MORPH_CLOSE, kernel, iterations=1)
opening_kernel4 = cv.morphologyEx(binr, cv.MORPH_OPEN, line_kernel4, iterations=1)
closing_kernel4 = cv.morphologyEx(opening_kernel4, cv.MORPH_CLOSE, kernel, iterations=1)
opening_kernel5 = cv.morphologyEx(binr, cv.MORPH_OPEN, line_kernel5, iterations=1)
closing_kernel5 = cv.morphologyEx(opening_kernel5, cv.MORPH_CLOSE, kernel, iterations=1)
opening_kernel6 = cv.morphologyEx(binr, cv.MORPH_OPEN, line_kernel6, iterations=1)
closing_kernel6 = cv.morphologyEx(opening_kernel6, cv.MORPH_CLOSE, kernel, iterations=1)
opening_kernel7 = cv.morphologyEx(binr, cv.MORPH_OPEN, line_kernel7, iterations=1)
closing_kernel7 = cv.morphologyEx(opening_kernel7, cv.MORPH_CLOSE, kernel, iterations=1)
opening_kernel8 = cv.morphologyEx(binr, cv.MORPH_OPEN, line_kernel8, iterations=1)
closing_kernel8 = cv.morphologyEx(opening_kernel8, cv.MORPH_CLOSE, kernel, iterations=1)
opening_kernel9 = cv.morphologyEx(binr, cv.MORPH_OPEN, line_kernel9, iterations=1)
closing_kernel9 = cv.morphologyEx(opening_kernel9, cv.MORPH_CLOSE, kernel, iterations=1)
opening_kernel10 = cv.morphologyEx(binr, cv.MORPH_OPEN, line_kernel10, iterations=1)
closing_kernel10 = cv.morphologyEx(opening_kernel10, cv.MORPH_CLOSE, kernel, iterations=1)
opening_kernel11 = cv.morphologyEx(binr, cv.MORPH_OPEN, line_kernel11, iterations=1)
closing_kernel11 = cv.morphologyEx(opening_kernel11, cv.MORPH_CLOSE, kernel, iterations=1)
opening_kernel12 = cv.morphologyEx(binr, cv.MORPH_OPEN, line_kernel12, iterations=1)
closing_kernel12 = cv.morphologyEx(opening_kernel12, cv.MORPH_CLOSE, kernel, iterations=1)


#combining the results of each kernel to make our lines
Lines = (closing_kernel1 + closing_kernel2 + closing_kernel3 + closing_kernel4 + closing_kernel5 + closing_kernel6 + 
closing_kernel7 + closing_kernel8 + closing_kernel9 + closing_kernel10 + closing_kernel11 + closing_kernel12)

#for accurate count we do the following
#to make one line out of the two segments of a line:
closing_kernel9 = cv.dilate(closing_kernel9,kernel,iterations=2)

closing_kernel7 = cv.erode(closing_kernel7,kernel,iterations=1)


#saving the results:
cv.imwrite('line.bmp',Lines)
cv.imwrite('circle.bmp',circles)

#saving the results of each kernel:
cv.imwrite('closing_kernel1.png' ,closing_kernel1)
cv.imwrite('closing_kernel2.png' ,closing_kernel2)
cv.imwrite('closing_kernel3.png' ,closing_kernel3)
cv.imwrite('closing_kernel4.png' ,closing_kernel4)
cv.imwrite('closing_kernel5.png' ,closing_kernel5)
cv.imwrite('closing_kernel6.png' ,closing_kernel6)
cv.imwrite('closing_kernel7.png' ,closing_kernel7)
cv.imwrite('closing_kernel8.png' ,closing_kernel8)
cv.imwrite('closing_kernel9.png' ,closing_kernel9)
cv.imwrite('closing_kernel10.png' ,closing_kernel10)
cv.imwrite('closing_kernel11.png' ,closing_kernel11)
cv.imwrite('closing_kernel12.png' ,closing_kernel12)


#counting the circles:

circle_image = cv.imread('circles.jpg', cv.IMREAD_GRAYSCALE)
ret, thresh = cv.threshold(circle_image,127,255,cv.THRESH_BINARY)
num_labels,lables = cv.connectedComponents(thresh)
print("number of connected components of circles : ",num_labels-1)

'''
#our first attempt to count the lines:

line_image = cv.imread('line.bmp', cv.IMREAD_GRAYSCALE)
ret_line, thresh_line = cv.threshold(line_image,127,255,cv.THRESH_BINARY)
num_labels_line,lables_line= cv.connectedComponents(thresh_line)
print("number of connected components of lines : ",num_labels_line-1)

'''
#counting the lines:

total_line_count = 0

line_kernel_1 = cv.imread("closing_kernel1.png", cv.IMREAD_GRAYSCALE)
ret1, thresh1 = cv.threshold(line_kernel_1,127,255,cv.THRESH_BINARY)
num_labels1,lables1 = cv.connectedComponents(thresh1)
total_line_count = total_line_count + num_labels1 - 1
#-1 is for deducting the background as a component


line_kernel_2 = cv.imread("closing_kernel2.png", cv.IMREAD_GRAYSCALE)
ret2, thresh2 = cv.threshold(line_kernel_2,127,255,cv.THRESH_BINARY)
num_labels2,lables2 = cv.connectedComponents(thresh2)
total_line_count = total_line_count + num_labels2 - 2
#the other -1 is because of the little line that appeared but is not a complete one in this kernel


line_kernel_3 = cv.imread("closing_kernel3.png", cv.IMREAD_GRAYSCALE)
ret3, thresh3 = cv.threshold(line_kernel_3,127,255,cv.THRESH_BINARY)
num_labels3,lables3 = cv.connectedComponents(thresh3)
total_line_count = total_line_count + num_labels3 - 1

line_kernel_4 = cv.imread("closing_kernel4.png", cv.IMREAD_GRAYSCALE)
ret4, thresh4 = cv.threshold(line_kernel_4,127,255,cv.THRESH_BINARY)
num_labels4,lables4 = cv.connectedComponents(thresh4)
total_line_count = total_line_count + num_labels4 - 1


line_kernel_5 = cv.imread("closing_kernel5.png", cv.IMREAD_GRAYSCALE)
ret5, thresh5 = cv.threshold(line_kernel_5,127,255,cv.THRESH_BINARY)
num_labels5,lables5 = cv.connectedComponents(thresh5)
total_line_count = total_line_count + num_labels5 - 1


line_kernel_6 = cv.imread("closing_kernel6.png", cv.IMREAD_GRAYSCALE)
ret6, thresh6 = cv.threshold(line_kernel_6,127,255,cv.THRESH_BINARY)
num_labels6,lables6 = cv.connectedComponents(thresh6)
total_line_count = total_line_count + num_labels6 - 1


line_kernel_7 = cv.imread("closing_kernel7.png", cv.IMREAD_GRAYSCALE)
ret7, thresh7 = cv.threshold(line_kernel_7,127,255,cv.THRESH_BINARY)
num_labels7,lables7 = cv.connectedComponents(thresh7)
total_line_count = total_line_count + num_labels7 - 1


line_kernel_8 = cv.imread("closing_kernel8.png", cv.IMREAD_GRAYSCALE)
ret8, thresh8 = cv.threshold(line_kernel_8,127,255,cv.THRESH_BINARY)
num_labels8,lables8 = cv.connectedComponents(thresh8)
total_line_count = total_line_count + num_labels8 - 1


line_kernel_9 = cv.imread("closing_kernel9.png", cv.IMREAD_GRAYSCALE)
ret9, thresh9 = cv.threshold(line_kernel_9,127,255,cv.THRESH_BINARY)
num_labels9,lables9 = cv.connectedComponents(thresh9)
total_line_count = total_line_count + num_labels9 - 1


line_kernel_10 = cv.imread("closing_kernel10.png", cv.IMREAD_GRAYSCALE)
ret10, thresh10 = cv.threshold(line_kernel_10,127,255,cv.THRESH_BINARY)
num_labels10,lables10 = cv.connectedComponents(thresh10)
total_line_count = total_line_count + num_labels10 - 1


line_kernel_11 = cv.imread("closing_kernel11.png", cv.IMREAD_GRAYSCALE)
ret11, thresh11 = cv.threshold(line_kernel_11,127,255,cv.THRESH_BINARY)
num_labels11,lables11 = cv.connectedComponents(thresh11)
total_line_count = total_line_count + num_labels11 - 1


line_kernel_12 = cv.imread("closing_kernel12.png", cv.IMREAD_GRAYSCALE)
ret12, thresh12 = cv.threshold(line_kernel_12,127,255,cv.THRESH_BINARY)
num_labels12,lables12 = cv.connectedComponents(thresh12)
total_line_count = total_line_count + num_labels12 - 1


print("number of lines : " , total_line_count)



#showing the results

plt.subplot(3,2,1)
plt.imshow(binr, cmap="gray")
plt.title('Binary Image')

plt.subplot(3,2,2)
plt.imshow(erosion, cmap="gray")
plt.title('Eroded Image')

plt.subplot(3,2,3)
plt.imshow(opening, cmap="gray")
plt.title('Opened Image')

plt.subplot(3,2,4)
plt.imshow(kernel , cmap="gray")
plt.title('Kernel')

plt.subplot(3,2,5)
plt.imshow(dilation, cmap="gray")
plt.title('Dilated Image')

plt.subplot(3,2,6)
plt.imshow(Lines,cmap="gray")
plt.title("lines")



'''

#showing the result of each line kernel:

plt.subplot(3,4,1)
plt.imshow(closing_kernel1,cmap="gray")
plt.title("line kernel 1")


plt.subplot(3,4,2)
plt.imshow(closing_kernel2,cmap="gray")
plt.title("line kernel 2")


plt.subplot(3,4,3)
plt.imshow(closing_kernel3,cmap="gray")
plt.title("line kernel 3")


plt.subplot(3,4,4)
plt.imshow(closing_kernel4,cmap="gray")
plt.title("line kernel 4")


plt.subplot(3,4,5)
plt.imshow(closing_kernel5,cmap="gray")
plt.title("line kernel 5")


plt.subplot(3,4,6)
plt.imshow(closing_kernel6,cmap="gray")
plt.title("line kernel 6")


plt.subplot(3,4,7)
plt.imshow(closing_kernel7,cmap="gray")
plt.title("line kernel 7")


plt.subplot(3,4,8)
plt.imshow(closing_kernel8,cmap="gray")
plt.title("line kernel 8")


plt.subplot(3,4,9)
plt.imshow(closing_kernel9,cmap="gray")
plt.title("line kernel 9")


plt.subplot(3,4,10)
plt.imshow(closing_kernel10,cmap="gray")
plt.title("line kernel 10")


plt.subplot(3,4,11)
plt.imshow(closing_kernel11,cmap="gray")
plt.title("line kernel 11")


plt.subplot(3,4,12)
plt.imshow(closing_kernel12,cmap="gray")
plt.title("line kernel 12")

'''

'''
#to show our line kernels :

plt.subplot(3,4,1)
plt.imshow(Line_kernel1,cmap="gray")
plt.title("line kernel 1")


plt.subplot(3,4,2)
plt.imshow(Line_kernel2,cmap="gray")
plt.title("line kernel 2")


plt.subplot(3,4,3)
plt.imshow(Line_kernel3,cmap="gray")
plt.title("line kernel 3")


plt.subplot(3,4,4)
plt.imshow(Line_kernel4,cmap="gray")
plt.title("line kernel 4")


plt.subplot(3,4,5)
plt.imshow(Line_kernel5,cmap="gray")
plt.title("line kernel 5")


plt.subplot(3,4,6)
plt.imshow(Line_kernel6,cmap="gray")
plt.title("line kernel 6")


plt.subplot(3,4,7)
plt.imshow(Line_kernel7,cmap="gray")
plt.title("line kernel 7")


plt.subplot(3,4,8)
plt.imshow(Line_kernel8,cmap="gray")
plt.title("line kernel 8")


plt.subplot(3,4,9)
plt.imshow(Line_kernel9,cmap="gray")
plt.title("line kernel 9")


plt.subplot(3,4,10)
plt.imshow(Line_kernel10,cmap="gray")
plt.title("line kernel 10")


plt.subplot(3,4,11)
plt.imshow(Line_kernel11,cmap="gray")
plt.title("line kernel 11")


plt.subplot(3,4,12)
plt.imshow(Line_kernel12,cmap="gray")
plt.title("line kernel 12")

'''

plt.tight_layout()
plt.show()
