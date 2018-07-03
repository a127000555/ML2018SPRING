file_name = ['outb467.csv','outb3.csv','sim3.csv']
import numpy as np 
ans = np.zeros([5060,6])

for f in file_name:
	with open(f,'r') as fin:
		first = True
		i = 0
		for row in fin:
			if first == True:
				first = False
				continue
			ans[i][int(row.strip().split(',')[1])] += 1

			i += 1

i = 0
with open('ans3.csv','w') as fout:
	print('id,ans',file=fout)
	for row in ans:
		print('{},{}'.format(i,np.argmax(row)),file=fout)
		i+=1
print(ans)