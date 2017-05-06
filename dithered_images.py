from __future__ import division 
import numpy as np
from numpy import sin, cos, pi, arange
import cv2
import matplotlib.pyplot as plt
import plotly.plotly as py

def create_sin_image(w, h, T, ph, avg, rng):
	img = np.zeros((h,w))
	x = arange(w)
	I_max = avg + rng/2
	y = np.array([ (1+(cos((2*pi*i/T) + ph)))/2 for i in arange(w)])
	img[:,:] = y
	return img

def create_sin_image2(w, h, f, ph, avg, rng):
	img = np.zeros((h,w))
	x = arange(w)
	y = np.array([ avg + rng*(cos((2*pi*f *(i/w)) + ph)) for i in arange(w)])
	img[:,:] = y
	return img

def sin_vari_freq():
	x = arange(500)
	f = arange(8,1,1/500)
	y = np.array([ 0.5 + 0.5*(cos((2*pi*f *(i/w)) + ph)) for i in arange(w)])
	return img

def create_dlp_bitmap(w,h,image):
	dlp_image = np.zeros((h,w,3))
	dlp_image[:,:,0] = dlp_image[:,:,1] = dlp_image[:,:,2] = image
	return dlp_image

def random_dithered(img):
	thresh = np.random.random((img.shape[0], img.shape[1]))
	dithered = np.zeros((img.shape[0], img.shape[1]))
	dithered[img>thresh] = 1
	return dithered

def binary_threshed(img):
	thresh = np.zeros((img.shape[0], img.shape[1]))
	thresh[:,:] = 0.5
	binarised = np.zeros((img.shape[0], img.shape[1]))
	binarised[img>thresh] = 1
	return binarised

def binary_vari_thresh(img, t):
	thresh = np.zeros((img.shape[0], img.shape[1]))
	thresh[:,:] = t
	binarised = np.zeros((img.shape[0], img.shape[1]))
	binarised[img>thresh] = 1
	return binarised

def error_diffused(img):
	a = np.zeros((img.shape[0], img.shape[1]))
	a[:,:] = img[:,:]
	s = img.shape
	b = np.zeros((img.shape[0], img.shape[1]))
	for i in range(s[0]):
		for j in range(s[1]):
			if (a[i,j] < 0.5):
				b[i,j] = 0
			else:
				b[i,j] = 1
			qerror = a[i,j] - b[i,j]
			if (j<(s[1]-1)):
				a[i,j+1] =  ((7/16*qerror)+a[i,j+1])
			if(i<(s[0]-1) and j>1):
				a[(i+1),(j-1)] = (a[(i+1),(j-1)] + (3/16*qerror))
				a[i+1,j] = (a[i+1,j] + (5/16*qerror))
			if(j<(s[1]-1) and i<(s[0]-1)):
				a[i+1,j+1] = (a[i+1,j+1] + (qerror)/16)
	a = np.round(a)
	return a

def low_pass(img, side):
	kernel = np.ones((side, side),np.float32)/(side*side)
	dst = cv2.filter2D(img,-1,kernel)
	return dst

def plot_dith_imgs(img, type, s1, s2):
	dith_type = {
		'Binary' : binary_threshed(img),
		'Random Threshold' : random_dithered(img),
		'Error Diffusion' : error_diffused(img)
	}
	dith 		= dith_type[type]
	dith_lp1 	= low_pass(dith, s1)[10,:]
	dith_lp2 	= low_pass(dith, s2)[10,:]
	fringe 		= img[10,:]
	dith 		= dith[10,:]
	fig = plt.figure()
	title = "Ideal Fringe vs " + type + " Dithered"
	fig.suptitle(title)
	plt.subplot(311)
	plt.xlabel('x index')
	plt.ylabel('Intensity')
	plt.ylim((-0.2,1.1))
	I1, = plt.plot(fringe, '--', label = 'Fringe Pattern')
	s 	= "Unfiltered Pattern"
	I2, = plt.plot(dith, 'r', label = s)
	plt.legend(handles=[I1, I2])
	plt.subplot(312)
	plt.xlabel('x index')
	plt.ylabel('Intensity')
	plt.ylim((-0.2,1.1))
	I1, = plt.plot(fringe, '--', label = 'Fringe Pattern')
	s 	= "Kernal Size: " + str(s1)
	I2, = plt.plot(dith_lp1, 'r', label = s)
	plt.legend(handles=[I1, I2])
	plt.subplot(313)
	plt.xlabel('x index')
	plt.ylabel('Intensity')
	plt.ylim((-0.2,1.1))
	I1, = plt.plot(fringe, '--', label = 'Fringe Pattern')
	s 	= "Kernal Size: " + str(s2)
	I2, = plt.plot(dith_lp2, 'r', label = s)
	plt.legend(handles=[I1, I2])

def plot_dithered_errors(img):
	# create dithered images
	thr = binary_threshed(img)
	ran = random_dithered(img)
	dif = error_diffused(img)
	thr_error = [255*sum(sum(abs(img-low_pass(thr, i))))/(image.shape[0]*image.shape[1]) for i in (1+np.arange(20))]
	thr_range = [255*(max(low_pass(thr, i)[10,:]) - min(low_pass(thr, i)[10,:])) for i in (1+np.arange(20))]
	ran_error = [255*sum(sum(abs(img-low_pass(ran, i))))/(image.shape[0]*image.shape[1]) for i in (1+np.arange(20))]
	ran_range = [255*(max(low_pass(ran, i)[10,:]) - min(low_pass(ran, i)[10,:])) for i in (1+np.arange(20))]
	dif_error = [255*sum(sum(abs(img-low_pass(dif, i))))/(image.shape[0]*image.shape[1]) for i in (1+np.arange(20))]
	dif_range = [255*(max(low_pass(dif, i)[10,:]) - min(low_pass(dif, i)[10,:])) for i in (1+np.arange(20))]
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	title = "Mean Pixel Error and Dynamic Range with Increasing Defocus"
	fig.suptitle(title)
	#plt.subplot(421)
	ax1.set_xlabel('low pass filter kernal size')
	ax1.set_ylabel('mean absolute error')
	ax2.set_ylabel('dynamic range')
	plt.xlim((-1,25))
	#plt.ylim((-0.1, 1.1))
	I1, = ax1.plot(thr_error, 'b', label = 'Binary Error')
	I2, = ax1.plot(ran_error, 'g', label = 'Random Error')
	I3, = ax1.plot(dif_error, 'r', label = 'Diffused Error')
	I4, = ax2.plot(thr_range, 'b--', label = 'Binary Range')
	I5, = ax2.plot(ran_range, 'g--', label = 'Random Range')
	I6, = ax2.plot(dif_range, 'r--', label = 'Diffused Range')
	plt.legend(handles=[I1,I2,I3,I4,I5,I6])
	#fig = plt.figure(2)
	#title = "Dynamic Range with Increasing Defocus"
	#fig.suptitle(title)
	#I1, = plt.plot(thr_range, label = 'Binary Error')
	#I2, = plt.plot(ran_range, label = 'Random Error')
	#I3, = plt.plot(dif_range, label = 'Diffused Error')
	#plt.legend(handles=[I1,I2,I3])
	plt.show()

def simple_thresh_example():
	image = cv2.imread('C:/FYP/Pytests/napkin.png')
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255

	bin1 = binary_vari_thresh(image, 0.5)
	bin2 = binary_vari_thresh(image, 0.75)
	ran  = random_dithered(image)
	while(1):
		cv2.imshow('grayscale', image)
		cv2.imshow('thresh = 0.5', bin1)
		cv2.imshow('thresh = 0.55', bin2)
		cv2.imshow('randomly dithered', ran)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

def lena_examples():
	img = cv2.imread('C:/FYP/Pytests/Lena.png')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = img/255
	img_defoc = low_pass(img, 6)
	binary = binary_threshed(img)
	r_dith = random_dithered(img)
	r_defoc = low_pass(r_dith, 6)
	e_diff = error_diffused(img)
	e_defoc = low_pass(e_diff, 6)
	while(1):
		cv2.imshow('Lena', img)
		#cv2.imshow('Lena Defoc', img_defoc)
		cv2.imshow('Binary', binary)
		cv2.imshow('Randomly Dithered', r_dith)
		#cv2.imshow('Defocused Rand', r_defoc)
		cv2.imshow('Error Diffusion', e_diff)
		#cv2.imshow('Defocused Erro Diff', e_defoc)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

def kernal_example():
	original = np.zeros((300,300))
	black = np.zeros((300,300))
	diff = np.zeros((300,300))
	original[:,:] = 4/9
	original[99,:] = original[199,:] = black[99,:] = black[199,:] = diff[99,:] = diff[199,:] =1
	original[:,99] = original[:,199] = black[:,99] = black[:,199] = diff[:,99] = diff[:,199] = 1
	diff[100:200,0:100] = 1
	diff[0:100,100:200] = 1
	diff[100:200,200:-1] = 1
	diff[200:-1,100:200] = 1
	diff[:,0] = 1
	all_images = np.concatenate((original,black,diff),axis=1)

	while(1):
		#cv2.imshow('original',original)
		#cv2.imshow('simple threshold',black)
		#cv2.imshow('error diffused',diff)
		cv2.imshow('kernal example',all_images)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == "__main__":
	noise = np.round(0.6*np.random.random((100, 320)))/5
	noise2 = np.round(0.6*np.random.random((100, 320)))/10
	fringe1 = create_sin_image2(320, 100, 10, 0, 0.5, 0.5)
	fringe1n = create_sin_image2(320, 100, 10, 0, 0.5, 0.5)
	fringe1n += noise2
	fringe2 = create_sin_image2(320, 100, 10, 0, 0.5, 0.35)
	fringe2n = create_sin_image2(320, 100, 10, 0, 0.5, 0.35)
	fringe2n += noise2
	fringe1 = np.concatenate((fringe1,fringe1n),axis=1)
	fringe2 = np.concatenate((fringe2,fringe2n),axis=1)
	noise = np.concatenate((noise,noise),axis=1)
	fringe = np.concatenate((noise,fringe1,fringe2),axis=0)
	"""img = cv2.imread('C:/FYP/Pytests/cameraman.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = img/255
	img_3 = low_pass(img, 3)
	img_6 = low_pass(img, 6)

	img = np.concatenate((img,img_3,img_6),axis=1)
	"""
	"""
	#simple_thresh_example()
	#lena_examples()
	#kernal_example()
	#fringe = create_sin_image2(640, 300, 20, 0, 0.5, 0.5)
	fringe0 = create_dlp_bitmap(912, 1140, np.round(random_dithered(create_sin_image(912, 1140, 15, -2*pi/3, 0.5, 0.5))*255))
	fringe1 = create_dlp_bitmap(912, 1140, np.round(random_dithered(create_sin_image(912, 1140, 15, 0, 0.5, 0.5))*255))
	fringe2 = create_dlp_bitmap(912, 1140, np.round(random_dithered(create_sin_image(912, 1140, 15, 2*pi/3, 0.5, 0.5))*255))
	#plot_dithered_errors(image)
	
	#plot_dith_imgs(image, 'Binary', 10, 20)
	#plot_dith_imgs(image, 'Random Threshold', 2, 7)
	#plot_dith_imgs(image, 'Error Diffusion', 2, 7)
	#plt.show()

	file = r"C:\\FYP\\Pytests\\Images\\T15\\Random\\pat0.bmp"
	print file
	cv2.imwrite(file, fringe0)
	file = r"C:\\FYP\\Pytests\\Images\\T15\\Random\\pat1.bmp"
	print file
	cv2.imwrite(file, fringe1)
	file = r"C:\\FYP\\Pytests\\Images\\T15\\Random\\pat2.bmp"
	print file
	cv2.imwrite(file, fringe2)
	
	#binary = binary_threshed(fringe)
	#random = random_dithered(fringe)
	#diffused = error_diffused(fringe)

	#binary = low_pass(binary, 20)
	#random = low_pass(random, 20)
	#diffused = low_pass(diffused, 20)
	"""

	while(1):
		cv2.imshow('Full Range vs Half Range', fringe)
		#cv2.imshow('Half Range', fringe2)
		#cv2.imshow('Binarised Image', binary)
		#cv2.imshow('Randomly Dithered Image', random)
		#cv2.imshow('Error Diffused Image', diffused)
		#cv2.imshow('Original Image', img)
		#cv2.imshow('Filtered Image', img_defoc)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	