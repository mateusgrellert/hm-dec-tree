import numpy
import os
import re
import string
import sys
import math

def appendParams(metrics, defParam):
  param = []
  for metric in metrics:
  	param.append(metric[defParam])
  return param

def calcMinMaxIntervals(metric_1, metric_2):
  intervals = []
  #calc min interval
  intervals.append(max([min(metric_1), min(metric_2)]))
  #calc max interval
  intervals.append(min([max(metric_1), max(metric_2)]))
  return intervals

def calcLogs(bitRates):
  setLogs = []
  for bitRate in bitRates:
  	setLogs.append(math.log10(bitRate))
  return setLogs

def pchipend(H0, H1, del0, del1):
	d = 0.0
	d = ((2*H0 + H1) * del0 - H0*del1)/(H0+H1)
	if(d*del0 < 0):
		d = 0.0
	elif ((del0 * del1 < 0) and (abs(d) > abs(3*del0))):
		d = 3*del0
	return d

def bdrint(log_rate, psnr, min_int, max_int):
	H = [0.0,0.0,0.0]
	delta = [0.0,0.0,0.0]
	d = [0.0,0.0,0.0,0.0]
	b = [0.0,0.0,0.0]
	c = [0.0,0.0,0.0]
	s0 = 0.0
	s1 = 0.0
	result = 0.0
	
	log_rate.reverse()
	psnr.reverse()
	
	
	for i in range(0,3):
		H[i] = psnr[i+1] - psnr[i]
		delta[i] = (log_rate[i+1] - log_rate[i])/H[i]
	
	d[0] = pchipend(H[0], H[1], delta[0], delta[1])
	for i in range(1,3):
		d[i] = (3*H[i-1] + 3*H[i])/((2*H[i]+H[i-1])/delta[i-1]+(H[i]+2*H[i-1])/delta[i])
		
	d[3] = pchipend(H[2], H[1], delta[2], delta[1])

	for i in range(0,3):
		c[i] = (3*delta[i] - 2*d[i] - d[i+1])/H[i]
		b[i] = (d[i] - 2*delta[i] + d[i+1])/(H[i]*H[i])
	
	
	for i in range(0,3):
		s0 = psnr[i]
		s1 = psnr[i+1]
		
		s0 = max(s0, min_int)
		s0 = min(s0, max_int)
		s1 = max(s1, min_int)
		s1 = min(s1, max_int)
		
		s0 = s0 - psnr[i]
		s1 = s1 - psnr[i]

		if(s1 > s0):
			result = result + (s1 - s0)*log_rate[i]
			result = result + (s1*s1-s0*s0)*d[i]/2
			result = result + (s1*s1*s1 - s0*s0*s0)*c[i]/3
			result = result + (s1*s1*s1*s1 - s0*s0*s0*s0)*b[i]/4
	return result
		





def bdrate(metrics_1, metrics_2):
	yuv_res = []
	for yuv_sel in range(len(metrics_1[0][1:])):
		bitRate_1 = []
		bitRate_2 = []
		PSNR_1 	= []
		PSNR_2 	= []

		log_1		= []
		log_2		= []
		intervals = []
		MAX_EXP	= 200

		bitRate_1 = appendParams(metrics_1, 0)
		PSNR_1 = appendParams(metrics_1, yuv_sel+1)
		bitRate_2 = appendParams(metrics_2, 0)
		PSNR_2 = appendParams(metrics_2, yuv_sel+1)

		#Calculate min and max intervals ([0] = min, [1] = max)
		intervals = calcMinMaxIntervals(PSNR_1, PSNR_2)

		diff_interval = intervals[1]-intervals[0]

		log_1 = calcLogs(bitRate_1)
		log_2 = calcLogs(bitRate_2)

		integr_1 = bdrint(log_1, PSNR_1, intervals[0], intervals[1])
		integr_2 = bdrint(log_2, PSNR_2, intervals[0], intervals[1])

		avgDiff = (integr_2-integr_1)/(diff_interval)

		if avgDiff > MAX_EXP :
			yuv_res.append(MAX_EXP)
		else:
			yuv_res.append(math.pow(10, avgDiff)-1)

	return yuv_res
