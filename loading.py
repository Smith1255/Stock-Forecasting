# import itertools
# import threading
# import time
# import sys

# def animate():
#     for c in itertools.cycle(['|', '/', '-', '\\']):
#         if done:
#             break
#         sys.stdout.write('\rloading ' + c)
#         sys.stdout.flush()
#         time.sleep(0.1)
#     sys.stdout.write('\rDone!     ')

# def start_loading():
# 	done = False
# 	t = threading.Thread(target=animate)
# 	t.start()

# def stop_loading():
# 	done = True

# start_loading()
# time.sleep(5)
# stop_loading()

import sys, time, threading
import os
def the_process_function():
	my_pid = os.getpid()
	print("Executing our Task on Process {}".format(my_pid))
	n = 20
	for i in range(n):
		time.sleep(0.5)
		sys.stdout.write('\r'+'loading...  process '+str(i)+'/'+str(n)+' '+ '{:.2f}'.format(i/n*100)+'%')
		sys.stdout.flush()
	sys.stdout.write('\r'+'loading... finished               \n')

def animated_loading():
	chars = "/—\|" 
	for char in chars:
		sys.stdout.write('\r'+'loading...'+char)
		time.sleep(.1)
		sys.stdout.flush() 

# the_process = threading.Thread(name='process', target=the_process_function)

# the_process.daemon = True
# the_process.start()

# while the_process.is_alive():
#     animated_loading()

import concurrent.futures

with concurrent.futures.ThreadPoolExecutor() as executor:
	future = executor.submit(the_process_function)
	while(future.running()):
		animated_loading()
